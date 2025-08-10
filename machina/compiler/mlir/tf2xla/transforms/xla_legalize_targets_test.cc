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

#include "machina/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace hlo {
namespace {

mlir::DialectRegistry GetDefaultDialectRegistry() {
  mlir::DialectRegistry registry;

  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<TF::TensorFlowDialect>();
  registry.insert<chlo::ChloDialect>();

  return registry;
}

class XlaLegalizeTargetsTest : public testing::Test {
 public:
  XlaLegalizeTargetsTest()
      : context_(GetDefaultDialectRegistry()),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_))),
        builder_(&module_->getBodyRegion()) {
    context_.loadAllAvailableDialects();
  }

 protected:
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::OpBuilder builder_;
};

TEST_F(XlaLegalizeTargetsTest, CreatesConversionTargets) {
  auto const_int = builder_.create<mlir::arith::ConstantIntOp>(
      builder_.getUnknownLoc(), builder_.getI32Type(), /*value=*/10);

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/false);
  EXPECT_TRUE(target.isLegal(const_int));
}

TEST_F(XlaLegalizeTargetsTest, AllowsCHLODialect) {
  auto const_int = builder_.create<chlo::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getI32TensorAttr({42}));

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/true);

  EXPECT_TRUE(target.isIllegal(const_int));
}

TEST_F(XlaLegalizeTargetsTest, DontAllowCHLODialect) {
  auto const_int = builder_.create<chlo::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getI32TensorAttr({42}));

  ConversionTarget target =
      GetDefaultLegalConversionTargets(context_, /*legalize_chlo=*/false);
  EXPECT_TRUE(target.isLegal(const_int));
}

}  // namespace
}  // namespace hlo
}  // namespace mlir
