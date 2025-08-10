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

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tf2xla/internal/passes/lowering_passes.h"
#include "machina/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/lib/monitoring/cell_reader.h"

namespace machina {
namespace tf2xla {
namespace internal {

namespace {

using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::hlo::test::GetMlirModuleFromString;
using ::machina::monitoring::testing::CellReader;

constexpr char kNotDynamicFunctionName[] = "kNotDynamicFunction";
constexpr char kDynamicFunctionName[] = "kDynamicFunction";
static constexpr char kDynamismOpCounterStreamzName[] =
    "/machina/core/tf2xla/api/v2/dynamism_op_counter";
static constexpr char kDynamismFunctionCounterStreamzName[] =
    "/machina/core/tf2xla/api/v2/dynamism_function_counter";

class InputLoweringMetricsPassTest : public testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));
    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addNestedPass<mlir::func::FuncOp>(CreateInputLoweringMetricsPass());
  }

  bool ModulesEqual(const ModuleOp& module_before,
                    const ModuleOp& module_after) {
    return mlir::OperationEquivalence::isEquivalentTo(
        module_before, module_after, mlir::OperationEquivalence::None);
  }

  mlir::LogicalResult Run() {
    mlir::OwningOpRef<mlir::ModuleOp> module_before = module_->clone();
    LogicalResult run_result = pm_->run(module_.get());

    EXPECT_TRUE(ModulesEqual(*module_before, *module_));
    return run_result;
  }

 private:
  mlir::MLIRContext context_;
  mlir::OwningOpRef<ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(InputLoweringMetricsPassTest, CountsNoDynamicOps) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      return %0 : tensor<1xi32>
    }
  })";

  CellReader<int64_t> dynamism_op_counter(kDynamismOpCounterStreamzName);
  CellReader<int64_t> dynamism_function_counter(
      kDynamismFunctionCounterStreamzName);

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(dynamism_function_counter.Delta(kNotDynamicFunctionName), 1);
}

TEST_F(InputLoweringMetricsPassTest, CountsDynamicOps) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> () {
      %cst0 = "tf.Const"(){ value = dense<0> : tensor<3x5xi1>} : () -> tensor<3x5xi1>
      %0 = "tf.Where"(%cst0) : (tensor<3x5xi1>) -> tensor<?x2xi64>
      func.return
    }
  })";

  CellReader<int64_t> dynamism_counter(kDynamismOpCounterStreamzName);
  CellReader<int64_t> dynamism_function_counter(
      kDynamismFunctionCounterStreamzName);

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(dynamism_counter.Delta("tf.Where"), 1);
  EXPECT_EQ(dynamism_function_counter.Delta(kDynamicFunctionName), 1);
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
