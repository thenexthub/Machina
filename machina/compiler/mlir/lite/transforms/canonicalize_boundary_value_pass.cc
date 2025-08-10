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

#include "machina/compiler/mlir/lite/transforms/canonicalize_boundary_value_pass.h"

#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/lite/utils/utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

#define DEBUG_TYPE "canonicalize-boundary-value"

// Clamp constant -Inf/Inf to MIN/MAX float value.
template <typename OpTy>
struct ClampInfToMinMaxFloat : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy const_op,
                                PatternRewriter& rewriter) const override {
    Attribute attr = const_op.getValueAttr();
    if (auto float_attr = toolchain::dyn_cast<FloatAttr>(attr)) {
      if (float_attr.getValue().isInfinity()) {
        FloatType float_type = toolchain::dyn_cast<FloatType>(const_op.getType());
        if (!float_type) return failure();
        rewriter.replaceOpWithNewOp<OpTy>(
            const_op, rewriter.getFloatAttr(
                          float_type, APFloat::getLargest(
                                          float_type.getFloatSemantics(),
                                          float_attr.getValue().isNegative())));
        return success();
      }
    }

    ElementsAttr tensor_attr = toolchain::dyn_cast<ElementsAttr>(attr);
    if (!tensor_attr) return failure();

    Type type = tensor_attr.getType();
    ShapedType tensor_type = toolchain::cast<ShapedType>(type);
    auto float_type = dyn_cast<FloatType>(tensor_type.getElementType());
    if (!float_type) return failure();

    auto vals_orig = tensor_attr.getValues<APFloat>();
    // If all values are finite, no need to rewrite.
    if (toolchain::all_of(vals_orig, [&](APFloat val) { return !val.isInfinity(); }))
      return failure();

    SmallVector<APFloat> vals_new(toolchain::map_range(vals_orig, [&](APFloat val) {
      return val.isInfinity()
                 ? APFloat::getLargest(float_type.getFloatSemantics(),
                                       val.isNegative())
                 : val;
    }));
    rewriter.replaceOpWithNewOp<OpTy>(
        const_op, DenseElementsAttr::get(tensor_type, vals_new));
    return success();
  }
};
}  // end namespace

void CanonicalizeBoundaryValuePass::runOnOperation() {
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ClampInfToMinMaxFloat<stablehlo::ConstantOp>,
               ClampInfToMinMaxFloat<TF::ConstOp>,
               ClampInfToMinMaxFloat<arith::ConstantOp>>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // end namespace TFL
}  // end namespace mlir
