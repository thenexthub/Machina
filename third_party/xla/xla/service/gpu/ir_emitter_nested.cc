/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "machina/xla/service/gpu/ir_emitter_nested.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/Support/Casting.h"
#include "machina/xla/codegen/emitters/computation_fingerprint.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/hlo/ir/hlo_opcode.h"
#include "machina/xla/literal.h"
#include "machina/xla/service/gpu/ir_emission_utils.h"
#include "machina/xla/service/gpu/ir_emitter.h"
#include "machina/xla/service/gpu/ir_emitter_context.h"
#include "machina/xla/service/llvm_ir/buffer_assignment_util.h"
#include "machina/xla/service/llvm_ir/ir_array.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/service/llvm_ir/loop_emitter.h"
#include "machina/xla/service/llvm_ir/tuple_ops.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/status_macros.h"
#include "machina/xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class IrEmitterNested : public IrEmitter {
 public:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);

  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  absl::Status HandleParameter(HloInstruction* parameter) override;

  // Generate the code for the computation passed in the constructor, if it
  // wasn't already generated previously.
  // As well as generting the code for the function, emits code for global
  // constants, and also populates related information to 'ir_emitter_context_'
  // for large-constant initializations. Large constants don't get initializers
  // in the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  //
  // The allocation index for these constants will always be -1 (i.e. doesn't
  // correspond to any allocation)
  absl::StatusOr<toolchain::Function*> CodegenNestedComputation();

 protected:
  absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& element_generator) override;

 private:
  // Emits constants to generated LLVM IR, and also populates related
  // information to 'ir_emitter_context_' for large-constant initializations.
  absl::Status EmitConstants(const HloComputation& computation);

  const HloComputation& nested_computation_;
};

IrEmitterNested::IrEmitterNested(const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context,
                /*is_nested=*/true),
      nested_computation_(nested_computation) {}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
absl::StatusOr<toolchain::Function*> IrEmitterNested::CodegenNestedComputation() {
  // Include a fingerprint of the HLO in the function name. Currently, codegen
  // is invoked on temporary HLO objects, which means the address of the
  // computation is not necessarily unique.
  std::string fingerprint =
      emitters::GetComputationFingerprint(&nested_computation_, {});
  size_t hash = absl::Hash<std::string>{}(fingerprint);
  std::string function_name = llvm_ir::SanitizeFunctionName(
      absl::StrCat(nested_computation_.name(), "_",
                   absl::Hex(reinterpret_cast<intptr_t>(&nested_computation_)),
                   "_", absl::Hex(hash)));

  auto* function =
      ir_emitter_context_->llvm_module()->getFunction(function_name);
  if (function) return function;

  TF_RETURN_IF_ERROR(EmitConstants(nested_computation_));
  std::vector<const HloInstruction*> io_hlos;
  std::vector<toolchain::Type*> argument_types;
  std::vector<int64_t> argument_dereferenceable_bytes;
  const auto& params = nested_computation_.parameter_instructions();
  const auto n = params.size() + 1;
  io_hlos.reserve(n - 1);
  argument_types.reserve(n);
  argument_dereferenceable_bytes.reserve(n);
  for (const HloInstruction* param : params) {
    io_hlos.push_back(param);
    const Shape& param_shape = param->shape();
    argument_types.push_back(b_.getPtrTy());
    int64_t param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }

  const HloInstruction* root = nested_computation_.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(b_.getPtrTy());
    int64_t root_size = llvm_ir::ByteSizeOf(
        root_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }

  toolchain::FunctionType* function_type =
      toolchain::FunctionType::get(b_.getVoidTy(), argument_types, false);
  function = toolchain::Function::Create(
      function_type,                       // The function type.
      toolchain::GlobalValue::InternalLinkage,  // The linkage type.
      function_name,
      ir_emitter_context_->llvm_module());  // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64_t arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableParamAttr(arg_no, arg_size);
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  toolchain::BasicBlock* entry_bb =
      toolchain::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  toolchain::ReturnInst* ret_instr =
      toolchain::ReturnInst::Create(function->getContext(), entry_bb);
  b_.SetInsertPoint(ret_instr);

  std::vector<const HloInstruction*> non_io_hlos;
  non_io_hlos.push_back(root);
  for (const auto* hlo : nested_computation_.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation_.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(io_hlos, non_io_hlos);

  TF_RETURN_IF_ERROR(nested_computation_.root_instruction()->Accept(this));
  b_.SetInsertPoint(ret_instr);

  // Function epilogue: copy the output value back.
  {
    // TODO(cheshire) Duplication vs. EmitThreadLocalFunctionEpilogue
    const HloInstruction* root_instruction =
        nested_computation_.root_instruction();
    toolchain::Value* root_value = bindings_.GetBasePointer(*root_instruction);
    const Shape& return_shape = root_instruction->shape();

    // Last argument is the out parameter.
    toolchain::Argument* out_parameter = std::prev(function->arg_end(), 1);

    if (ShapeUtil::IsScalar(return_shape)) {
      toolchain::Value* ret_value =
          Load(llvm_ir::ShapeToIrType(return_shape, module_->getContext()),
               root_value, "load_ret_value");
      Store(ret_value, out_parameter);
    } else {
      CHECK(return_shape.IsTuple());
      toolchain::Type* tuple_type =
          llvm_ir::ShapeToIrType(return_shape, module_->getContext());

      for (int i = 0; i < return_shape.tuple_shapes().size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        toolchain::Value* destination = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, out_parameter, tuple_type, &b_);
        toolchain::Value* source = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, root_value,
            llvm_ir::ShapeToIrType(root_instruction->shape(),
                                   module_->getContext()),
            &b_);
        Store(Load(llvm_ir::ShapeToIrType(element_shape, module_->getContext()),
                   source),
              destination);
      }
    }
  }
  b_.SetInsertPoint(ret_instr);
  return function;
}

absl::Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status IrEmitterNested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  // For MOF we give the loop emitter an array for every output it should
  // generate.
  if (hlo.shape().IsTuple()) {
    std::vector<llvm_ir::IrArray> target_arrays =
        ConstructIrArrayForOutputs(hlo);
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, target_arrays, &b_).EmitLoop());
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), target_arrays, &b_);
    return absl::OkStatus();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

absl::Status IrEmitterNested::EmitConstants(const HloComputation& computation) {
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() != HloOpcode::kConstant) {
      continue;
    }
    const Literal& literal = instr->literal();

    // These globals will be looked up by name by GpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR (e.g. TupleThunk) so we can't give then a linkage that
    // merely preserves their names (like available_externally), we also need
    // to ensure that they stick around even if they're "unused".
    //
    // We may have to be more clever here in the future if we notice that we're
    // keeping around too many globals because of their linkage.
    std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);

    auto base = static_cast<const uint8_t*>(literal.untyped_data());
    ir_emitter_context_->emit_constant(
        literal.element_count(),
        ShapeUtil::ByteSizeOfPrimitiveType(literal.shape().element_type()),

        global_name,
        /*allocation_idx=*/-1,
        DenseDataIntermediate::Alias(
            absl::MakeSpan(base, base + literal.size_bytes())),
        &b_);
  }
  return absl::OkStatus();
}

// Casts the provided toolchain::Value* to the default address space. This is useful
// in particular for generating IR for AMDGPU target, as its kernel variables
// are in address space 5 instead of the default address space.
toolchain::Value* AddrCastToDefault(toolchain::Value* arg, toolchain::IRBuilderBase& b) {
  toolchain::Type* arg_type = arg->getType();
  CHECK(arg_type->isPointerTy());
  if (arg_type->getPointerAddressSpace() != 0) {
    toolchain::Type* generic_arg_type = toolchain::PointerType::get(
        toolchain::cast<toolchain::PointerType>(arg_type)->getContext(), 0);
    toolchain::Value* addrspacecast_arg =
        b.CreateAddrSpaceCast(arg, generic_arg_type);
    return addrspacecast_arg;
  }
  return arg;
}

}  // namespace

absl::Status CallNestedComputation(toolchain::IRBuilderBase* builder,
                                   IrEmitterContext& ir_emitter_context,
                                   const HloComputation& computation,
                                   absl::Span<toolchain::Value* const> operands,
                                   toolchain::Value* output) {
  TF_RET_CHECK(computation.num_parameters() > 0);

  TF_ASSIGN_OR_RETURN(toolchain::Function * emitted_function,
                      IrEmitterNested(computation, &ir_emitter_context)
                          .CodegenNestedComputation());

  // Operands are in default address space for non-AMDGPU target.
  // However for AMDGPU target, addrspacecast alloca variables from
  // addrspace 5 to addrspace 0 is needed.
  std::vector<toolchain::Value*> arguments;
  absl::c_transform(
      operands, std::back_inserter(arguments),
      [builder](toolchain::Value* arg) { return AddrCastToDefault(arg, *builder); });

  toolchain::Value* casted_output = AddrCastToDefault(output, *builder);
  arguments.push_back(casted_output);

  builder->CreateCall(emitted_function, arguments);

  return absl::OkStatus();
}

absl::StatusOr<std::vector<toolchain::Value*>> CallNestedComputationWithScalars(
    toolchain::IRBuilderBase* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<toolchain::Value* const> parameter_elements) {
  std::vector<toolchain::Value*> parameter_buffers;
  for (toolchain::Value* parameter_element : parameter_elements) {
    parameter_buffers.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        parameter_element->getType(), "parameter_buffer", builder));
    builder->CreateStore(parameter_element, parameter_buffers.back());
  }

  return CallNestedComputationWithScalarAddrs(builder, ir_emitter_context,
                                              computation, parameter_buffers);
}

absl::StatusOr<std::vector<toolchain::Value*>> CallNestedComputationWithScalarAddrs(
    toolchain::IRBuilderBase* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<toolchain::Value* const> parameter_elements_addrs) {
  const Shape& return_shape = computation.root_instruction()->shape();
  toolchain::Type* return_buffer_type =
      llvm_ir::ShapeToIrType(return_shape, builder->getContext());
  toolchain::Value* return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      return_buffer_type, "return_buffer", builder);

  std::vector<toolchain::Value*> allocas_for_returned_scalars;
  if (!return_shape.IsTuple()) {
    allocas_for_returned_scalars.push_back(return_buffer);
  } else {
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, builder);
    llvm_ir::IrArray tuple_array(return_buffer, return_buffer_type,
                                 return_shape);

    llvm_ir::EmitTuple(tuple_array, allocas_for_returned_scalars, builder);
  }

  TF_RETURN_IF_ERROR(
      CallNestedComputation(builder, ir_emitter_context, computation,
                            parameter_elements_addrs, return_buffer));

  std::vector<toolchain::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (toolchain::Value* addr : allocas_for_returned_scalars) {
    auto alloca = toolchain::cast<toolchain::AllocaInst>(addr);
    returned_scalars.push_back(
        builder->CreateLoad(alloca->getAllocatedType(), alloca));
  }
  return returned_scalars;
}

}  // namespace gpu
}  // namespace xla
