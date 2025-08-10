/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

// This transformation pass propagates QSV information through the model.

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/transforms/quantization/quant_utils.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_BIASQUANTIZERPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Returns the quantization dimension for the bias.
//
// Some kernels allow 1D bias, broadcasting it inside the kernel. In this case,
// the `quantizedDimension` is 0 for per-channel quantization. However, for
// some kernels that require the bias to be already broadcasted to match the
// accumulation shape, the very last dimension should be used. This function
// returns the last dimension of the bias.
//
// Returns -1 for scalar bias. Returns std::nullopt if bias is none.
std::optional<int> GetAdjustedQuantizationDimension(
    RequiresQuantizedBiasInterface op, int bias_index) {
  int adjusted_quant_dim = -1;
  if (op->getNumOperands() > bias_index) {
    const mlir::Value bias = op->getOperand(bias_index);
    const Type bias_type = bias.getType();
    if (mlir::isa<NoneType>(bias_type)) {
      return std::nullopt;
    }

    if (const auto shaped_bias_type =
            mlir::dyn_cast<mlir::ShapedType>(bias_type)) {
      if (shaped_bias_type.hasRank()) {
        const int bias_rank = shaped_bias_type.getRank();
        adjusted_quant_dim = bias_rank > 1 ? bias_rank - 1 : 0;
      }
    }
  }
  return adjusted_quant_dim;
}

std::optional<quant::QuantizedType> GetBiasQuantizedType(
    RequiresQuantizedBiasInterface op) {
  int bias_index = op.GetBiasOperandIndex();
  std::vector<int> non_bias_operand_indices = op.GetNonBiasOperandIndices();

  std::optional<int> adjusted_quant_dim_opt =
      GetAdjustedQuantizationDimension(op, bias_index);
  if (!adjusted_quant_dim_opt) {
    return std::nullopt;
  }

  std::vector<QuantizedType> op_types{};
  op_types.reserve(non_bias_operand_indices.size());
  for (const int non_bias_operand_index : non_bias_operand_indices) {
    auto operand_qtype =
        GetQTypeFromDefiningDequantize(op->getOperand(non_bias_operand_index));
    if (operand_qtype.has_value()) {
      op_types.push_back(operand_qtype.value());
    }
  }
  if (op_types.size() < non_bias_operand_indices.size()) {
    // Not all the non-bias operands are quantized so not quantizing the bias.
    // This could for example happen in weight-only.
    return std::nullopt;
  }
  return mlir::TFL::GetUniformQuantizedTypeForBias(op_types,
                                                   *adjusted_quant_dim_opt);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class QuantizeBias
    : public OpInterfaceRewritePattern<RequiresQuantizedBiasInterface> {
  using OpInterfaceRewritePattern<
      RequiresQuantizedBiasInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::RequiresQuantizedBiasInterface op,
                                PatternRewriter& rewriter) const override {
    auto new_bias_qtype = GetBiasQuantizedType(op);
    if (!new_bias_qtype) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Failed to get bias quantized type");
    }
    auto bias_value = op->getOperand(op.GetBiasOperandIndex());
    auto existing_bias_qtype = GetQTypeFromDefiningDequantize(bias_value);
    if (existing_bias_qtype && *existing_bias_qtype == *new_bias_qtype) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Bias already quantized with the same type");
    }

    // Provide the target_op to ensure QDQ is only inserted for this use.
    if (failed(InsertQDQ(bias_value, *new_bias_qtype, rewriter,
                         op.getOperation()))) {
      op->emitError("Failed to insert QDQ before bias");
      return failure();
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct BiasQuantizerPass
    : public impl::BiasQuantizerPassBase<BiasQuantizerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BiasQuantizerPass)

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void BiasQuantizerPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeBias>(ctx);

  GreedyRewriteConfig greedy_config;
  greedy_config.enableFolding(true);

  if (failed(
          applyPatternsGreedily(module, std::move(patterns), greedy_config))) {
    module.emitError("Failed to apply BiasQuantizerPass patterns.");
    signalPassFailure();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateBiasQuantizerPass() {
  return std::make_unique<BiasQuantizerPass>();
}

}  // namespace TFL
}  // namespace mlir
