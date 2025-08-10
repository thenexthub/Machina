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

#include <utility>

#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/passes/quantization_patterns.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_QUANTIZEPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Base struct for quantization.
template <typename ConcreteT,
          typename RootOpT = mlir::quant::ir::DequantizeCastOp>
struct StableHloQuantizationBase
    : public StableHloQuantizationPattern<ConcreteT,
                                          mlir::quant::ir::QuantizeCastOp,
                                          mlir::quant::ir::DequantizeCastOp,
                                          /*VerifierT=*/void, RootOpT> {
  explicit StableHloQuantizationBase(MLIRContext* ctx)
      : StableHloQuantizationPattern<ConcreteT, mlir::quant::ir::QuantizeCastOp,
                                     mlir::quant::ir::DequantizeCastOp,
                                     /*VerifierT=*/void, RootOpT>(ctx) {}

  static bool AllowWeightOnlyQuantization(Operation& op) { return false; }
};

// Quantization rewrite pattern using DQ as the root op.
struct StableHloQuantization
    : public StableHloQuantizationBase<StableHloQuantization> {
  explicit StableHloQuantization(MLIRContext* ctx)
      : StableHloQuantizationBase<StableHloQuantization>(ctx) {}
};

// Quantization rewrite pattern using Q as the root op. This is for the
// quantizable ops without floating-point operands.
struct StableHloQuantizationReverse
    : public StableHloQuantizationBase<StableHloQuantizationReverse,
                                       mlir::quant::ir::QuantizeCastOp> {
  explicit StableHloQuantizationReverse(MLIRContext* ctx)
      : StableHloQuantizationBase<StableHloQuantizationReverse,
                                  mlir::quant::ir::QuantizeCastOp>(ctx) {}
};

class QuantizePass : public impl::QuantizePassBase<QuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)

  using impl::QuantizePassBase<QuantizePass>::QuantizePassBase;

  explicit QuantizePass(const bool enable_per_channel_quantized_weight) {
    enable_per_channel_quantized_weight_ = enable_per_channel_quantized_weight;
  }

 private:
  void runOnOperation() override;
};

void QuantizePass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<StableHloQuantization, StableHloQuantizationReverse>(&ctx);

  PopulateCommonQuantizationPatterns(ctx, patterns,
                                     enable_per_channel_quantized_weight_);

  // Quantize all quantizable ops, including ops that are not compute-heavy.
  PopulateAllQuantizablePatterns(ctx, patterns);

  if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
    // There are cases where no rewrites happen even if a pattern matches,
    // causing this to result in a convergence failure. Consider this as a
    // best-effort.
    module_op.emitWarning("Failed to converge pattern at QuantizePass.");
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo
