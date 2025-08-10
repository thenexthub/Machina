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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo {
namespace {

#define GEN_PASS_DEF_OPTIMIZEINTGRAPH
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

class OptimizeIntGraph : public impl::OptimizeIntGraphBase<OptimizeIntGraph> {
 public:
  OptimizeIntGraph() = default;
  OptimizeIntGraph(const OptimizeIntGraph &) = default;
  void runOnOperation() override;
};

#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.inc"

void OptimizeIntGraph::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  auto func = getOperation();
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeIntGraphPass() {
  return std::make_unique<OptimizeIntGraph>();
}

}  // namespace mlir::quant::stablehlo
