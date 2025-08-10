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

#include "machina/compiler/mlir/quantization/stablehlo/utils/stablehlo_type_utils.h"

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::quant::stablehlo {
namespace {

using ::testing::Test;

class StablehloTypeUtilsTest : public Test {
 protected:
  StablehloTypeUtilsTest() {
    ctx_.loadDialect<mlir::stablehlo::StablehloDialect,
                     mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_{&ctx_};
};

TEST_F(StablehloTypeUtilsTest, IsStablehloOpSucceedsWithStablehloOp) {
  const OwningOpRef<mlir::stablehlo::ConstantOp> constant_op =
      builder_.create<mlir::stablehlo::ConstantOp>(
          builder_.getUnknownLoc(), builder_.getI32IntegerAttr(0));
  EXPECT_TRUE(IsStablehloOp(*constant_op));
}

TEST_F(StablehloTypeUtilsTest, IsStablehloOpFailsWithArithOp) {
  const OwningOpRef<mlir::arith::ConstantOp> constant_op =
      builder_.create<mlir::arith::ConstantOp>(builder_.getUnknownLoc(),
                                               builder_.getI32IntegerAttr(0));
  EXPECT_FALSE(IsStablehloOp(*constant_op));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
