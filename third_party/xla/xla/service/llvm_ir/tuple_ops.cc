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

#include "machina/xla/service/llvm_ir/tuple_ops.h"

#include <stddef.h>

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Module.h"
#include "machina/xla/service/llvm_ir/ir_array.h"
#include "machina/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/xla_data.pb.h"

namespace xla {
namespace llvm_ir {

static toolchain::Module* getModuleFromBuilder(toolchain::IRBuilderBase* b) {
  return b->GetInsertBlock()->getModule();
}

void EmitTuple(const IrArray& tuple, absl::Span<toolchain::Value* const> operands,
               toolchain::IRBuilderBase* b) {
  for (size_t i = 0; i < operands.size(); ++i) {
    auto* cast = b->CreatePointerCast(
        operands[i], PrimitiveTypeToIrType(TUPLE, b->getContext()));
    auto* store = b->CreateStore(
        cast,
        b->CreateInBoundsGEP(tuple.GetBasePointeeType(), tuple.GetBasePointer(),
                             {b->getInt64(0), b->getInt64(i)}));
    tuple.AnnotateLoadStoreInstructionWithMetadata(store);
  }
}

void EmitTuple(const IrArray& tuple, absl::Span<const IrArray> buffers,
               toolchain::IRBuilderBase* b) {
  std::vector<toolchain::Value*> buffer_ptrs;
  buffer_ptrs.reserve(buffers.size());
  absl::c_transform(
      buffers, std::back_inserter(buffer_ptrs),
      [](const llvm_ir::IrArray& buffer) { return buffer.GetBasePointer(); });
  llvm_ir::EmitTuple(tuple, buffer_ptrs, b);
}

std::vector<toolchain::Value*> EmitTupleAllocasAtFunctionEntry(
    const Shape& tuple_shape, toolchain::IRBuilderBase* b) {
  toolchain::IRBuilderBase::InsertPointGuard guard(*b);
  toolchain::Function* function = b->GetInsertBlock()->getParent();
  b->SetInsertPoint(&function->getEntryBlock(),
                    function->getEntryBlock().getFirstInsertionPt());
  CHECK(tuple_shape.IsTuple());
  int tuple_size = tuple_shape.tuple_shapes().size();

  std::vector<toolchain::Value*> generated_allocas;
  for (int i = 0; i < tuple_size; i++) {
    const Shape& element_shape = tuple_shape.tuple_shapes(i);
    CHECK(ShapeUtil::IsScalar(element_shape));
    toolchain::Type* type = llvm_ir::PrimitiveTypeToIrType(
        element_shape.element_type(), b->getContext());
    toolchain::AllocaInst* alloca = b->CreateAlloca(
        type,
        /*ArraySize=*/nullptr, AsStringRef(absl::StrCat("tuple_element_", i)));
    generated_allocas.push_back(alloca);
  }

  return generated_allocas;
}

toolchain::Value* EmitGetTupleElement(const Shape& target_shape, int64_t index,
                                 int alignment, toolchain::Value* operand,
                                 toolchain::Type* operand_pointee_type,
                                 toolchain::IRBuilderBase* b) {
  const std::vector<toolchain::Value*> gep_index = {b->getInt64(0),
                                               b->getInt64(index)};
  toolchain::Value* element_ptr =
      b->CreateInBoundsGEP(operand_pointee_type, operand, gep_index);
  toolchain::Type* element_pointee_type =
      toolchain::GetElementPtrInst::getIndexedType(operand_pointee_type, gep_index);
  toolchain::LoadInst* src_buffer = b->CreateLoad(element_pointee_type, element_ptr);

  // Mark the loaded pointer as dereferenceable if we know its shape.
  if (!target_shape.IsOpaque()) {
    SetDereferenceableMetadataForLoad(
        src_buffer,
        ByteSizeOf(target_shape, src_buffer->getModule()->getDataLayout()));
  }
  SetAlignmentMetadataForLoad(src_buffer, alignment);
  return src_buffer;
}

}  // namespace llvm_ir
}  // namespace xla
