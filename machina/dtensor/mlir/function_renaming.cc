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

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "toolchain/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/constants.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORFUNCTIONRENAMING
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

struct DTensorFunctionRenaming
    : public impl::DTensorFunctionRenamingBase<DTensorFunctionRenaming> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    const std::string append =
        module->getAttrOfType<mlir::StringAttr>(dtensor::kCacheKey)
            .getValue()
            .str();

    // If the cache key isn't set, simply return without renameing functions.
    if (append.empty()) return;

    mlir::SymbolTableCollection symbol_table;
    mlir::SymbolUserMap symbolUsers(symbol_table, module);

    for (mlir::func::FuncOp func_op :
         toolchain::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      // Only rename private functions, functions which are public (i.e. the
      // main function of the module), must have stable names since they are
      // public and may be used by other modules/pieces of code.
      if (func_op.getVisibility() != mlir::SymbolTable::Visibility::Private)
        continue;
      std::string new_name = absl::StrCat(
          mlir::SymbolTable::getSymbolName(func_op).getValue().str(), append);
      symbolUsers.replaceAllUsesWith(
          func_op, mlir::StringAttr::get(&getContext(), new_name));
      mlir::SymbolTable::setSymbolName(func_op, new_name);
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateFunctionRenamingPass() {
  return std::make_unique<DTensorFunctionRenaming>();
}

}  // namespace dtensor
}  // namespace machina
