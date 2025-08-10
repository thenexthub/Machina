/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/compiler/mlir/lite/transforms/cleanup_optimization_barrier_pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace TFL {
namespace {

#define DEBUG_TYPE "cleanup-optimization-barrier"

// Replaces the shlo.optimization_barrier op with its input.
struct CleanupOptimizationBarrier
    : public OpRewritePattern<stablehlo::OptimizationBarrierOp> {
  using OpRewritePattern<stablehlo::OptimizationBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::OptimizationBarrierOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};
}  // end namespace

void CleanupOptimizationBarrierPass::runOnOperation() {
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<CleanupOptimizationBarrier>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // end namespace TFL
}  // end namespace mlir
