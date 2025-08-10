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

#include "machina/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"

namespace machina {
namespace tf2xla {
namespace internal {

namespace {

using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::UnknownLoc;
using mlir::chlo::ChloDialect;
using mlir::TF::TensorFlowDialect;
using machina::tf2xla::internal::IsInBridgeAcceptableDialects;

class SharedUtilsTest : public ::testing::Test {};

TEST_F(SharedUtilsTest, IsInFunctionalDialectPasses) {
  MLIRContext context;
  context.loadDialect<TensorFlowDialect>();
  OpBuilder opBuilder(&context);
  OperationState state(UnknownLoc::get(opBuilder.getContext()),
                       /*OperationName=*/"tf.Const");
  mlir::Operation* op = Operation::create(state);

  bool result = IsInBridgeAcceptableDialects(op);

  EXPECT_TRUE(result);
  op->destroy();
}

TEST_F(SharedUtilsTest, IsInFunctionalDialectFails) {
  MLIRContext context;
  context.loadDialect<ChloDialect>();
  OpBuilder opBuilder(&context);
  OperationState state(UnknownLoc::get(opBuilder.getContext()),
                       /*OperationName=*/"chlo.broadcast_add");
  Operation* op = Operation::create(state);

  bool result = IsInBridgeAcceptableDialects(op);

  EXPECT_FALSE(result);
  op->destroy();
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
