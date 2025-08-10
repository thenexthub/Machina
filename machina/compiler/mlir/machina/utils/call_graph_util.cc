/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include <vector>

#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {

std::vector<toolchain::StringRef> GetEntryFunctionAttributeNames() {
  return {"tf.entry_function",
          tf_saved_model::kTfSavedModelInitializerTypeAttr};
}

bool IsEntryFunction(func::FuncOp func) {
  for (const auto &attr : GetEntryFunctionAttributeNames()) {
    if (func->hasAttr(attr)) {
      return true;
    }
  }
  return false;
}

toolchain::SmallVector<func::FuncOp> GetEntryFunctions(ModuleOp module) {
  toolchain::SmallVector<func::FuncOp> entry_funcs;
  module.walk([&](func::FuncOp func) {
    // A model may have multiple graphs, with each graph having its own entry.
    // When a graph is imported to MLIR, `tf.entry_function` will be added to
    // each entry function. The one exception are initializer functions, which
    // have `tf_saved_model.initializer_type` instead.
    if (IsEntryFunction(func)) {
      entry_funcs.push_back(func);
    }
  });
  return entry_funcs;
}

LogicalResult GetCallees(SymbolUserOpInterface op, SymbolTable &symtab,
                         toolchain::SmallVector<func::FuncOp> &callees) {
  for (auto attr : op->getAttrs()) {
    auto sym = mlir::dyn_cast<SymbolRefAttr>(attr.getValue());
    if (!sym) continue;
    auto callee = symtab.lookup<func::FuncOp>(sym.getRootReference());
    if (!callee) {
      // This is not expected to happen in practice.
      return op->emitError()
             << "Cannot find function " << sym.getRootReference();
    }
    callees.push_back(callee);
  }
  return success();
}

bool HasSingleBlock(func::FuncOp func) {
  return func->getNumRegions() == 1 && func.getBody().hasOneBlock();
}

}  // namespace mlir
