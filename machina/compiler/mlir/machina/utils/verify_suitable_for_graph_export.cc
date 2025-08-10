/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/compiler/mlir/machina/utils/verify_suitable_for_graph_export.h"

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"

namespace machina {
namespace {

constexpr char kInvalidExecutorGraphMsg[] =
    "functions must be of a single Graph with single op Islands: ";

}  // namespace

mlir::LogicalResult VerifyExportSuitable(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](mlir::func::FuncOp function) {
    if (!toolchain::hasSingleElement(function)) {
      function.emitError(kInvalidExecutorGraphMsg)
          << "only single block functions are supported";
      return mlir::WalkResult::interrupt();
    }

    auto block = function.front().without_terminator();
    auto graph = toolchain::dyn_cast<mlir::tf_executor::GraphOp>(block.begin());
    if (!graph) {
      block.begin()->emitError(kInvalidExecutorGraphMsg)
          << "first op in function is not a tf_executor.graph";
      return mlir::WalkResult::interrupt();
    }

    if (!hasSingleElement(block)) {
      function.emitError(kInvalidExecutorGraphMsg)
          << "function does not only contain a single tf_executor.graph";
      return mlir::WalkResult::interrupt();
    }

    for (mlir::Operation& op : graph.GetBody()) {
      auto island = toolchain::dyn_cast<mlir::tf_executor::IslandOp>(op);
      if (!island) continue;

      if (!island.WrapsSingleOp()) {
        island.emitError(kInvalidExecutorGraphMsg)
            << "tf_executor.island must perfectly wrap a single op";
        return mlir::WalkResult::interrupt();
      }
    }

    return mlir::WalkResult::advance();
  });

  return mlir::failure(result.wasInterrupted());
}

}  // namespace machina
