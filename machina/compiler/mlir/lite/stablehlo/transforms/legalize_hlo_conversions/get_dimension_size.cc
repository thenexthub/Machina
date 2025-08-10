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

#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/get_dimension_size.h"

#include <cstdint>

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/ImplicitLocOpBuilder.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// Converts a MHLO::GetDimensionSizeOP to TFL ops.
class LeagalizeDimensionSizeOp
    : public OpConversionPattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GetDimensionSizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto operand_type = toolchain::cast<ShapedType>(op.getOperand().getType());

    auto shaped_op_type =
        RankedTensorType::get({operand_type.getRank()}, rewriter.getI64Type());
    Value shape_op = TFL::ShapeOp::create(rewriter, op.getLoc(), shaped_op_type,
                                          op.getOperand());

    Value size = BuildIntArrayConstOp<arith::ConstantOp>(builder, rewriter, {1},
                                                         rewriter.getI64Type());

    auto begin = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter,
        toolchain::SmallVector<int64_t>({static_cast<int64_t>(op.getDimension())}),
        rewriter.getI64Type());

    auto slice_type = RankedTensorType::get({1}, rewriter.getI64Type());
    Value slice = TFL::SliceOp::create(rewriter, op.getLoc(), slice_type,
                                       shape_op, begin, size);

    auto op_el_type = toolchain::cast<ShapedType>(op.getType()).getElementType();
    if (op_el_type != slice_type.getElementType()) {
      slice = TFL::CastOp::create(rewriter, op->getLoc(),
                                  slice_type.clone(op_el_type), slice);
    }

    rewriter.replaceOpWithNewOp<TFL::SqueezeOp>(op, op.getType(), slice,
                                                rewriter.getI64ArrayAttr({0}));

    return success();
  }
};

}  // namespace

void PopulateGetDimensionSizePatterns(MLIRContext* ctx,
                                      RewritePatternSet& patterns,
                                      ConversionTarget& target) {
  target.addIllegalOp<mhlo::GetDimensionSizeOp>();
  patterns.add<LeagalizeDimensionSizeOp>(ctx);
}

}  // namespace mlir::odml
