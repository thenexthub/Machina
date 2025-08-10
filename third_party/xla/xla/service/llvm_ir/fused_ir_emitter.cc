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

#include "machina/xla/service/llvm_ir/fused_ir_emitter.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/hlo/ir/hlo_opcode.h"
#include "machina/xla/service/elemental_ir_emitter.h"
#include "machina/xla/service/llvm_ir/ir_array.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/util.h"

namespace xla {

using llvm_ir::IrArray;

absl::StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::DefaultAction(
    const HloInstruction& instruction) {
  IndexedGenerator generator = elemental_emitter_.MakeElementGenerator(
      &instruction, indexed_generators_);

  return absl::StatusOr<IndexedGenerator>([&, generator = std::move(generator)](
                                              const IrArray::Index& index)
                                              -> absl::StatusOr<toolchain::Value*> {
    ValueCacheKey key{&instruction, index.multidim()};
    toolchain::Value* value = value_cache_.insert({key, nullptr}).first->second;

    if (value != nullptr) {
      if (const auto* generated_instruction =
              toolchain::dyn_cast<toolchain::Instruction>(value)) {
        const toolchain::BasicBlock* bb = generated_instruction->getParent();

        // Ideally, we should be able to reuse the cached generated value if it
        // dominates the current insertion block. However, the check for
        // dominance can be expensive and unreliable when the function is being
        // constructed.
        //
        // It's also worth experimenting what if we don't do caching at all.
        // LLVM's CSE or GVN should be able to easily merge common
        // subexpressions that would be regenerated without caching. But this
        // might increase the JIT compilation time.
        toolchain::IRBuilderBase* b = elemental_emitter_.b();

        if (bb == b->GetInsertBlock()) {
          VLOG(3) << "The cached generated value is reused.";
          return value;
        }

        VLOG(3)
            << "The cached generated value can't be reused, because it is in "
               "a different BB ("
            << bb->getName().str() << ") from the current insertion block ("
            << b->GetInsertBlock()->getName().str() << ").";
      }
    }

    TF_ASSIGN_OR_RETURN(value, generator(index));
    value_cache_[std::move(key)] = value;
    return value;
  });
}

FusedIrEmitter::IndexedGenerator FusedIrEmitter::HandleConstant(
    const HloInstruction& constant) {
  toolchain::Module* module = elemental_emitter_.module();
  toolchain::IRBuilderBase* b = elemental_emitter_.b();

  // Explicitly set global addrspace for SPIR backend.
  int addrspace = toolchain::Triple(module->getTargetTriple()).isSPIR() ? 1 : 0;
  toolchain::Constant* initializer =
      llvm_ir::ConvertLiteralToIrConstant(constant.literal(), module);
  toolchain::GlobalVariable* global = new toolchain::GlobalVariable(
      *b->GetInsertBlock()->getModule(), initializer->getType(),
      /*isConstant=*/true,
      /*Linkage=*/toolchain::GlobalValue::PrivateLinkage,
      /*Initializer=*/initializer,
      /*Name=*/"", /*InsertBefore=*/nullptr,
      /*TLMode=*/toolchain::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/addrspace,
      /*isExternallyInitialized=*/false);
  global->setUnnamedAddr(toolchain::GlobalVariable::UnnamedAddr::Global);

  toolchain::Type* shape_type =
      llvm_ir::ShapeToIrType(constant.shape(), module->getContext());
  IrArray array(global, shape_type, constant.shape());

  return [&, b, array = std::move(array)](const IrArray::Index& index) {
    return array.EmitReadArrayElement(index, b, constant.name());
  };
}

absl::StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::HandleTuple(
    const HloInstruction& tuple) {
  std::vector<toolchain::Type*> element_ir_types;
  element_ir_types.reserve(tuple.operand_count());
  for (const HloInstruction* operand : tuple.operands()) {
    element_ir_types.push_back(llvm_ir::PrimitiveTypeToIrType(
        operand->shape().element_type(),
        elemental_emitter_.module()->getContext()));
  }

  toolchain::IRBuilderBase* b = elemental_emitter_.b();
  toolchain::Type* type = toolchain::StructType::get(b->getContext(), element_ir_types);

  return absl::StatusOr<IndexedGenerator>([&, b,
                                           type](const IrArray::Index& index)
                                              -> absl::StatusOr<toolchain::Value*> {
    toolchain::Value* ret = toolchain::UndefValue::get(type);
    for (size_t i = 0; i < tuple.operand_count(); ++i) {
      IrArray::Index used_index = index;
      if (i > 0 && !ShapeUtil::EqualIgnoringElementType(
                       tuple.operand(i)->shape(), tuple.operand(0)->shape())) {
        used_index = used_index.SourceIndexOfBitcast(
            tuple.operand(0)->shape(), tuple.operand(i)->shape(), b);
      }
      TF_ASSIGN_OR_RETURN(toolchain::Value * value,
                          indexed_generators_.at(tuple.operand(i))(used_index));
      ret = b->CreateInsertValue(ret, value, i);
    }
    return ret;
  });
}

absl::StatusOr<FusedIrEmitter::IndexedGenerator>
FusedIrEmitter::CreateGenerator(const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    case HloOpcode::kConstant:
      return HandleConstant(instruction);
    case HloOpcode::kGetTupleElement:
      return Internal("Tuple parameters are not supported for fusion");
    case HloOpcode::kParameter:
      return InvalidArgument("Unbound parameter: %s", instruction.ToString());
    case HloOpcode::kTuple:
      return HandleTuple(instruction);
    default:
      return DefaultAction(instruction);
  }
}

absl::StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::GetGenerator(
    const HloInstruction& instruction) {
  std::vector<const HloInstruction*> stack = {&instruction};
  while (!stack.empty()) {
    const HloInstruction& instr = *stack.back();
    stack.pop_back();

    IndexedGenerator& indexed_generator = indexed_generators_[&instr];
    if (indexed_generator != nullptr) continue;

    stack.insert(stack.end(), instr.operands().begin(), instr.operands().end());
    TF_ASSIGN_OR_RETURN(indexed_generator, CreateGenerator(instr));
  }
  return indexed_generators_[&instruction];
}

}  // namespace xla
