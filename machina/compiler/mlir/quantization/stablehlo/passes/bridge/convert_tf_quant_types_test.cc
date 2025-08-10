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

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/monitoring/cell_reader.h"

namespace mlir::quant::stablehlo {
namespace {

using ::mlir::DialectRegistry;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::machina::monitoring::testing::CellReader;
using ::testing::Test;

static constexpr char kMetricsName[] =
    "/machina/core/tf2xla/tf_quant_op_count";

class LegalizeTfTypesTest : public Test {
 protected:
  void CreateModule(const char* module_string) {
    DialectRegistry mlir_registry;
    RegisterCommonToolingDialects(mlir_registry);
    context_.appendDialectRegistry(mlir_registry);
    TF_ASSERT_OK(
        machina::DeserializeMlirModule(module_string, &context_, &module_));

    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addNestedPass<mlir::func::FuncOp>(
        quant::stablehlo::CreateConvertTFQuantTypesPass());
  }
  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(LegalizeTfTypesTest, RecordsStreamzQuantOps) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<3x3x!tf_type.qint8>, %arg1: tensor<3x3x!tf_type.qint8>) -> tensor<6x3x!tf_type.qint8> {
      %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
      %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3x!tf_type.qint8>, tensor<3x3x!tf_type.qint8>, tensor<i64>) -> tensor<6x3x!tf_type.qint8>
      func.return %1 : tensor<6x3x!tf_type.qint8>
    }
  })";
  CreateModule(kMlirModuleStr);
  CellReader<int64_t> reader(kMetricsName);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(reader.Delta("tf.ConcatV2"), 1);
  EXPECT_EQ(reader.Delta("func.return"), 1);
  EXPECT_EQ(reader.Delta("func.func"), 0);
}

TEST_F(LegalizeTfTypesTest, RecordsStreamzNoQuantOps) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
      %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
      %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
      func.return %1 : tensor<6x3xf32>
    }
  })";
  CreateModule(kMlirModuleStr);
  CellReader<int64_t> reader(kMetricsName);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(reader.Delta("tf.ConcatV2"), 0);
  EXPECT_EQ(reader.Delta("func.return"), 0);
  EXPECT_EQ(reader.Delta("func.func"), 0);
}

}  // namespace
}  // namespace mlir::quant::stablehlo
