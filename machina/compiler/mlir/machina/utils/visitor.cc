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

#include "machina/compiler/mlir/machina/utils/visitor.h"

#include <utility>

#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/STLFunctionalExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace mlir {
namespace TF {

WalkResult WalkReachableFunctions(
    func::FuncOp func,
    toolchain::function_ref<WalkResult(func::FuncOp)> callback,
    SymbolTableCollection* symbol_table) {
  toolchain::SmallDenseSet<Operation*> visited;

  toolchain::SmallVector<func::FuncOp> stack;
  stack.push_back(func);

  while (!stack.empty()) {
    func::FuncOp f = stack.back();
    stack.pop_back();

    if (!visited.insert(f).second) {
      continue;
    }

    WalkResult result = callback(f);
    if (result.wasInterrupted()) {
      return result;
    } else if (result.wasSkipped()) {
      continue;
    }

    result = f.walk([&](Operation* op) {
      const auto uses = SymbolTable::getSymbolUses(op);
      if (!uses.has_value()) {
        op->emitOpError() << "contains a potentially unknown symbol table";
        return WalkResult::interrupt();
      }

      for (const SymbolTable::SymbolUse& use : *uses) {
        func::FuncOp called_func =
            symbol_table != nullptr
                ? symbol_table->lookupNearestSymbolFrom<func::FuncOp>(
                      use.getUser(), use.getSymbolRef())
                : SymbolTable::lookupNearestSymbolFrom<
                      func::FuncOp>(use.getUser(), use.getSymbolRef());
        if (called_func == nullptr) {
          op->emitOpError()
              << "refers to an unknown symbol (expects a function)";
          return WalkResult::interrupt();
        }
        stack.push_back(called_func);
      }

      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return result;
    }
  }

  return WalkResult::advance();
}

FailureOr<OwningOpRef<ModuleOp>> CreatePrunedModule(
    ModuleOp module, toolchain::ArrayRef<toolchain::StringRef> function_names) {
  SymbolTableCollection symbol_table;
  OpBuilder builder(module.getContext());

  OwningOpRef<ModuleOp> pruned =
      builder.create<ModuleOp>(module->getLoc());
  (*pruned)->setAttrs(module->getAttrs());
  builder.setInsertionPointToEnd(pruned->getBody());

  toolchain::SmallDenseSet<func::FuncOp> added;
  for (const toolchain::StringRef function_name : function_names) {
    auto func =
        toolchain::dyn_cast_or_null<func::FuncOp>(symbol_table.lookupSymbolIn(
            module, builder.getStringAttr(function_name)));
    if (func == nullptr) {
      return module.emitError()
             << "Cannot find function '" << function_name << "'";
    }

    const WalkResult result = WalkReachableFunctions(
        func,
        [&](func::FuncOp f) {
          if (!added.insert(f).second) {
            return WalkResult::skip();
          }
          builder.clone(*f);
          return WalkResult::advance();
        },
        &symbol_table);
    if (result.wasInterrupted()) {
      return failure();
    }
  }

  return pruned;
}

}  // namespace TF
}  // namespace mlir
