/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Traits.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace {

class ConvertResultsBroadcastableShapeOp : public RewritePattern {
 public:
  ConvertResultsBroadcastableShapeOp(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 private:
  template <typename Op>
  LogicalResult RewriteEqOp(Operation* op, PatternRewriter& rewriter) const;

  LogicalResult RewriteOp(
      Operation* op, PatternRewriter& rewriter,
      const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                               SmallVectorImpl<int64_t>&)>&
          get_broadcasted_shape) const;

  LogicalResult RewriteBatchMatMulV2Op(Operation* op,
                                       PatternRewriter& rewriter) const;
};

#define GEN_PASS_DEF_BROADCASTFOLDPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class BroadcastFoldPass
    : public impl::BroadcastFoldPassBase<BroadcastFoldPass> {
 public:
  void runOnOperation() override;
};

LogicalResult ConvertResultsBroadcastableShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);

  // tf.Equal and tf.NotEqual ops only satisfy ResultsBroadcastableShape when
  // incompatible_shape_error is `true` (what is also checked by the verifier).
  if (succeeded(RewriteEqOp<TF::EqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteEqOp<TF::NotEqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteBatchMatMulV2Op(op, rewriter))) return success();

  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteBatchMatMulV2Op(
    Operation* op, PatternRewriter& rewriter) const {
  auto matmul_op = toolchain::dyn_cast<TF::BatchMatMulV2Op>(op);
  if (!matmul_op) return failure();

  // Gets the broadcasted output shape for tf.BatchMatMulV2Op. `shape_x` is the
  // shape of op's first/left-hand-side operand and `shape_y` is the shape of
  // op's second/right-hand-side operand.
  const auto get_broadcasted_shape =
      [&](ArrayRef<int64_t> shape_x, ArrayRef<int64_t> shape_y,
          SmallVectorImpl<int64_t>& result_shape) {
        if (shape_x.size() < 2 || shape_y.size() < 2) {
          return false;
        }

        // Checks outer dimensions (i.e., the dimensions higher than 2D) are
        // broadcastable. If true, then get the broadcasted shape for outer
        // dimension.
        if (!OpTrait::util::getBroadcastedShape(
                shape_x.drop_back(2), shape_y.drop_back(2), result_shape)) {
          return false;
        }

        const int x_row =
            matmul_op.getAdjX() ? shape_x.back() : *(shape_x.rbegin() + 1);
        const int x_col =
            !matmul_op.getAdjX() ? shape_x.back() : *(shape_x.rbegin() + 1);

        const int y_row =
            matmul_op.getAdjY() ? shape_y.back() : *(shape_y.rbegin() + 1);
        const int y_col =
            !matmul_op.getAdjY() ? shape_y.back() : *(shape_y.rbegin() + 1);

        // Checks that matrix multiply can perform a valid contraction.
        if (x_col != y_row) {
          result_shape.clear();
          return false;
        }

        result_shape.push_back(x_row);
        result_shape.push_back(y_col);
        return true;
      };

  return RewriteOp(op, rewriter, get_broadcasted_shape);
}

template <typename Op>
LogicalResult ConvertResultsBroadcastableShapeOp::RewriteEqOp(
    Operation* op, PatternRewriter& rewriter) const {
  auto eq_op = toolchain::dyn_cast_or_null<Op>(op);
  if (eq_op && eq_op.getIncompatibleShapeError())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);
  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteOp(
    Operation* op, PatternRewriter& rewriter,
    const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                             SmallVectorImpl<int64_t>&)>& get_broadcasted_shape)
    const {
  if (op->getNumOperands() != 2 || op->getResultTypes().size() != 1)
    return failure();

  // Check that the result shape is fully defined.
  auto result_type =
      mlir::dyn_cast_or_null<RankedTensorType>(op->getResultTypes().front());
  if (!result_type || !result_type.hasStaticShape()) return failure();

  bool changed = false;
  for (uint64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    // Check that the i'th operand is a broadcast.
    auto broadcast = toolchain::dyn_cast_or_null<TF::BroadcastToOp>(
        op->getOpOperand(i).get().getDefiningOp());
    if (!broadcast) continue;

    // Check that the operand of the broadcast has fully defined shape.
    auto broadcast_arg_type = mlir::dyn_cast_or_null<RankedTensorType>(
        broadcast.getInput().getType());
    if (!broadcast_arg_type || !broadcast_arg_type.hasStaticShape()) continue;

    // Check that the other argument has fully defined shape.
    auto argument_type = mlir::dyn_cast_or_null<RankedTensorType>(
        op->getOpOperand(1 - i).get().getType());
    if (!argument_type || !argument_type.hasStaticShape()) continue;

    // Get the unbroadcasted shapes in the operand order.
    std::array<toolchain::ArrayRef<int64_t>, 2> operand_shapes;
    operand_shapes[i] = broadcast_arg_type.getShape();
    operand_shapes[1 - i] = argument_type.getShape();

    // Check that the input of the broadcast and the other operand is broadcast
    // compatible.
    toolchain::SmallVector<int64_t, 4> broadcasted_shape;
    if (!get_broadcasted_shape(operand_shapes[0], operand_shapes[1],
                               broadcasted_shape))
      continue;

    // Check that an implicit broadcast between the operand of the broadcast and
    // the other argument would result in the same type as the result type.
    if (broadcasted_shape != result_type.getShape()) continue;

    // Update the operand of the op to be the operand of the broadcast.
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(i).set(broadcast.getInput()); });
    changed = true;
  }
  return success(changed);
}

void BroadcastFoldPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertResultsBroadcastableShapeOp>(func.getContext());
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<func::FuncOp>> CreateBroadcastFoldPass() {
  return std::make_unique<BroadcastFoldPass>();
}
}  // namespace TF

}  // namespace mlir
