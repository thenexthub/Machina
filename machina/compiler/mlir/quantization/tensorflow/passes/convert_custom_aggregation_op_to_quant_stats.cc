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
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/quantization/machina/passes/tf_quant_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class ConvertCustomAggregationOpToQuantStatsPass
    : public PassWrapper<ConvertCustomAggregationOpToQuantStatsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertCustomAggregationOpToQuantStatsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-convert-tf-custom-aggregator-op-to-quant-stats";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert tf.CustomAggregator op to quant.Stats";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<quant::QuantDialect>();
    registry.insert<mlir::quant::ir::TFQuantDialect>();
  }

  void runOnOperation() override;
};

class ConvertCustomAggregationOpToQuantStats
    : public OpRewritePattern<TF::CustomAggregatorOp> {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit ConvertCustomAggregationOpToQuantStats(MLIRContext *context)
      : OpRewritePattern<TF::CustomAggregatorOp>(context) {}

  LogicalResult matchAndRewrite(TF::CustomAggregatorOp op,
                                PatternRewriter &rewriter) const override {
    FloatAttr min = mlir::dyn_cast_or_null<FloatAttr>(op->getAttr("min"));
    FloatAttr max = mlir::dyn_cast_or_null<FloatAttr>(op->getAttr("max"));

    // When there are no min and max attributes, remove op.
    if (min == nullptr || max == nullptr) {
      op.getOutput().replaceAllUsesWith(op.getInput());
      rewriter.eraseOp(op);
      return success();
    }

    // The layer stats contain only the first min/max pairs.
    ElementsAttr layer_stats = DenseFPElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getF32Type()),
        {static_cast<float>(min.getValueAsDouble()),
         static_cast<float>(max.getValueAsDouble())});
    ElementsAttr axis_stats;
    IntegerAttr axis;

    mlir::quant::ir::StatisticsOp stats_op =
        rewriter.create<mlir::quant::ir::StatisticsOp>(
            op->getLoc(), op.getInput(), layer_stats, axis_stats, axis);
    op.getOutput().replaceAllUsesWith(stats_op.getResult());
    return success();
  }
};

static PassRegistration<ConvertCustomAggregationOpToQuantStatsPass> pass;

void ConvertCustomAggregationOpToQuantStatsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<ConvertCustomAggregationOpToQuantStats>(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    func.emitError()
        << "quant-convert-tf-custom-aggregator-op-to-quant-stats failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass() {
  return std::make_unique<ConvertCustomAggregationOpToQuantStatsPass>();
}

}  // namespace quant
}  // namespace mlir
