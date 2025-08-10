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

#include <iterator>
#include <memory>
#include <tuple>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"

namespace mlir {

namespace {

#define GEN_PASS_DEF_EXECUTORDIALECTTOFUNCTIONALPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct ExecutorDialectToFunctionalConversion
    : public impl::ExecutorDialectToFunctionalPassBase<
          ExecutorDialectToFunctionalConversion> {
  void runOnOperation() override;
};

// Extracts inner ops of tf_executor.island ops in a tf_executor.graph, in the
// order of ops in tf_executor.graph.
LogicalResult LiftIslandOpInnerOpsFromGraph(tf_executor::GraphOp graph) {
  auto graph_position = graph.getOperation()->getIterator();
  Block* parent_block = graph.getOperation()->getBlock();
  for (Operation& op : graph.GetBody().without_terminator()) {
    auto island_op = toolchain::dyn_cast<tf_executor::IslandOp>(op);
    if (!island_op)
      return op.emitOpError()
             << "is not supported for lifting out of tf_executor.graph, "
                "expected tf_executor.island";

    // Move inner ops in island to before the outer graph.
    auto& island_body = island_op.GetBody().getOperations();
    parent_block->getOperations().splice(graph_position, island_body,
                                         island_body.begin(),
                                         std::prev(island_body.end()));
    // Forward island fetches (tf_executor.yield operands) to island op result
    // uses.
    for (auto result :
         toolchain::zip(island_op.getOutputs(), island_op.GetYield().getFetches()))
      std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  // Forward graph fetches (tf_executor.fetch operands) to graph op result uses.
  for (auto result :
       toolchain::zip(graph.getResults(), graph.GetFetch().getFetches()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  graph.erase();
  return success();
}

void ExecutorDialectToFunctionalConversion::runOnOperation() {
  auto result = getOperation().walk([](tf_executor::GraphOp graph) {
    if (failed(LiftIslandOpInnerOpsFromGraph(graph)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (result.wasInterrupted()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateExecutorDialectToFunctionalConversionPass() {
  return std::make_unique<ExecutorDialectToFunctionalConversion>();
}

}  // namespace mlir
