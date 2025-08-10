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
#include "machina/compiler/mlir/lite/transforms/tflite_passes/unfold_large_splat_constants_pass.h"

#include <cstddef>

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h"
#include "machina/compiler/mlir/machina/utils/dynamic_shape_utils.h"

namespace mlir {
namespace TFL {
namespace {
// The threshold of constant bits to be unfolded (1Mb). If there is a splat
// constant with size equal or greater to this threshold, then it will be
// unfolded back to a regular `tfl.fill` operation.
constexpr size_t kConstantSizeThresholdInBits = 1e+6;
void MaybeUnfoldLargeSplatConstant(mlir::OpBuilder* op_builder,
                                   mlir::arith::ConstantOp const_op) {
  auto splat_elements_attr =
      mlir::dyn_cast<SplatElementsAttr>(const_op.getValue());
  if (!splat_elements_attr) {
    return;
  }
  auto element_type = splat_elements_attr.getType().getElementType();
  if (!(element_type.isF32() || element_type.isF16() ||
        element_type.isInteger(1) || element_type.isInteger(32) ||
        element_type.isInteger(64))) {
    return;
  }
  if (splat_elements_attr.getNumElements() *
          splat_elements_attr.getType().getElementTypeBitWidth() <
      kConstantSizeThresholdInBits) {
    return;
  }

  op_builder->setInsertionPoint(const_op);
  mlir::arith::ConstantOp fill_shape =
      op_builder->create<mlir::arith::ConstantOp>(
          const_op->getLoc(), DenseIntElementsAttr::get(
                                  machina::GetTypeFromTFTensorShape(
                                      {splat_elements_attr.getType().getRank()},
                                      op_builder->getI64Type()),
                                  splat_elements_attr.getType().getShape()));
  mlir::arith::ConstantOp fill_value =
      op_builder->create<mlir::arith::ConstantOp>(
          const_op->getLoc(),
          DenseElementsAttr::get(
              machina::GetTypeFromTFTensorShape(
                  {}, splat_elements_attr.getType().getElementType()),
              splat_elements_attr.getSplatValue<Attribute>()));
  TFL::FillOp fill = op_builder->create<TFL::FillOp>(
      const_op->getLoc(), splat_elements_attr.getType(), fill_shape,
      fill_value);
  const_op->replaceAllUsesWith(fill);
  const_op->erase();
}
}  // namespace

void UnfoldLargeSplatConstantPass::runOnOperation() {
  auto module = getOperation();

  mlir::OpBuilder op_builder(&module.getBodyRegion());
  module.walk([&](mlir::arith::ConstantOp const_op) {
    MaybeUnfoldLargeSplatConstant(&op_builder, const_op);
  });
}

}  // namespace TFL
}  // namespace mlir
