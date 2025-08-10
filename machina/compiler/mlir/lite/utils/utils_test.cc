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

#include "machina/compiler/mlir/lite/utils/utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {
namespace {

// Test fixture for AreBroadcastAndReductionAxesIndependent function.
class BroadcastAndReductionAxesIndependentTest : public ::testing::Test {
 protected:
  BroadcastAndReductionAxesIndependentTest() : builder_(&context_) {
    context_.loadDialect<arith::ArithDialect>();
  }

  // Builds an mlir::Value representing a tensor with the given shape.
  Value BuildTensor(ArrayRef<int64_t> shape) {
    return arith::ConstantOp::create(
        builder_, builder_.getUnknownLoc(),
        RankedTensorType::get(shape, builder_.getF32Type()),
        builder_.getZeroAttr(
            RankedTensorType::get(shape, builder_.getF32Type())));
  }

  // Builds a DenseElementsAttr representing an integer array.
  DenseElementsAttr BuildIntArrayAttr(ArrayRef<int32_t> values) {
    return DenseElementsAttr::get(
        RankedTensorType::get({static_cast<int32_t>(values.size())},
                              builder_.getI32Type()),
        values);
  }

  MLIRContext context_;
  OpBuilder builder_;
};

TEST_F(BroadcastAndReductionAxesIndependentTest, IndependentAxes) {
  Value input_tensor = BuildTensor({2, 1, 4, 1});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_TRUE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, OverlappingAxes) {
  Value input_tensor = BuildTensor({1, 3, 4, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, EmptyReductionAxes) {
  Value input_tensor = BuildTensor({1, 3, 1, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_TRUE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, UnrankedInput) {
  Value input_tensor = arith::ConstantOp::create(
      builder_, builder_.getUnknownLoc(), builder_.getF32Type(),
      builder_.getZeroAttr(builder_.getF32Type()));
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, InvalidReductionAxesType) {
  Value input_tensor = BuildTensor({2, 3, 4, 5});
  DenseElementsAttr reduction_axes = DenseElementsAttr::get(
      RankedTensorType::get({2}, builder_.getF32Type()), {1.0f, 2.0f});
  DenseElementsAttr target_shape = BuildIntArrayAttr({1, 3, 1, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, InvalidTargetShapeType) {
  Value input_tensor = BuildTensor({2, 3, 4, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = DenseElementsAttr::get(
      RankedTensorType::get({2}, builder_.getF32Type()), {1.0f, 2.0f});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

}  // namespace
}  // namespace TFL

}  // namespace mlir
