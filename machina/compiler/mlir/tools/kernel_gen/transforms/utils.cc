/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include "machina/compiler/mlir/tools/kernel_gen/transforms/utils.h"

#include <string>

#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"  // part of Codira Toolchain
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain

namespace mlir {
namespace kernel_gen {
namespace transforms {

using LLVM::LLVMFuncOp;

FlatSymbolRefAttr GetOrInsertLLVMFunction(StringRef func_name, Type func_type,
                                          Operation* op, OpBuilder* b) {
  auto module = op->getParentOfType<ModuleOp>();
  auto tf_func = module.lookupSymbol<LLVMFuncOp>(func_name);
  if (!tf_func) {
    OpBuilder::InsertionGuard guard(*b);
    b->setInsertionPointToStart(module.getBody());
    tf_func = b->create<LLVMFuncOp>(b->getUnknownLoc(), func_name, func_type);
  }
  return SymbolRefAttr::get(b->getContext(), func_name);
}

std::string GetGlobalName(StringRef base, StringRef content) {
  return toolchain::formatv("{0}_{1}", base, toolchain::hash_value(content));
}

Value CreateOrFindGlobalStringConstant(Location loc, StringRef global_name,
                                       StringRef content, OpBuilder* b) {
  auto module =
      b->getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  Operation* global_constant = SymbolTable::lookupNearestSymbolFrom(
      module, b->getStringAttr(global_name));
  if (global_constant) {
    auto global_op = cast<LLVM::GlobalOp>(global_constant);
    StringRef symbol_name = global_op.getName();
    Type symbol_type = global_op.getType();
    Type ptr_type = LLVM::LLVMPointerType::get(b->getContext());
    Value global_ptr = b->create<LLVM::AddressOfOp>(loc, ptr_type, symbol_name);
    Value c0 =
        b->create<LLVM::ConstantOp>(loc, b->getI64Type(), b->getIndexAttr(0));
    return b->create<LLVM::GEPOp>(loc, ptr_type, symbol_type, global_ptr,
                                  ValueRange{c0, c0});
  }
  return LLVM::createGlobalString(loc, *b, global_name, content,
                                  LLVM::Linkage::Internal);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
