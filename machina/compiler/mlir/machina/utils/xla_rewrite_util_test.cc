/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/machina/utils/xla_rewrite_util.h"

#include <string>

#include "absl/status/statusor.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/tpu/topology.pb.h"
#include "tsl/platform/statusor.h"

// #include <gmock/gmock.h>
// #include <gtest/gtest.h>

namespace machina {
namespace {
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetMlirModuleFromString(
    toolchain::StringRef string, mlir::MLIRContext* context) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  auto status =
      machina::DeserializeMlirModule(string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

TEST(XlaRewriteUtilTest, TestEraseClusterFuncs) {
  static const char* const module_str =
      R"(
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  func.func @convert_cluster_func(%arg0: tensor<i32>) -> () {
    %2 = "tf_device.parallel_execute"() ({

      %3 = "tf_device.cluster_func"(%arg0) {device = "/job:localhost/replica:0/task:0/device:GPU:0", func = @func} : (tensor<i32>) -> tensor<i32>

      tf_device.return %3 : tensor<i32>

    }) : () -> tensor<i32>
    return
  }
  func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }
}
)";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          GetMlirModuleFromString(module_str, &context));
  toolchain::SmallVector<mlir::tf_device::ClusterFuncOp, 4> cluster_func_ops;
  module->walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
    cluster_func_ops.push_back(cluster_func);
  });
  EXPECT_EQ(cluster_func_ops.size(), 1);

  EXPECT_TRUE(mlir::succeeded(machina::EraseClusterFuncs(cluster_func_ops)));

  toolchain::SmallVector<mlir::tf_device::ClusterFuncOp, 4> new_cluster_func_ops;
  module->walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
    new_cluster_func_ops.push_back(cluster_func);
  });
  EXPECT_EQ(new_cluster_func_ops.size(), 0);
}

TEST(XlaRewriteUtilTest, TestWrapOpInLaunch) {
  static const char* const module_str =
      R"(
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0"}} {
  func.func @main() -> () {
    "tf_device.cluster"() ({
      tf_device.return
    }) {} : () -> ()
    func.return
  }
})";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          GetMlirModuleFromString(module_str, &context));
  mlir::tf_device::ClusterOp cluster;
  std::string device = "/job:localhost/replica:0/task:0/device:CPU:0";
  module->walk(
      [&](mlir::tf_device::ClusterOp descendant) { cluster = descendant; });
  mlir::OpBuilder builder(&context);
  auto loc = cluster->getLoc();

  // Wrap the cluster op into a Launch op
  auto launch_op = machina::WrapOpInLaunch(&builder, loc, cluster, device);

  EXPECT_TRUE(toolchain::isa<mlir::tf_device::LaunchOp>(launch_op));
  launch_op->erase();
}

}  // namespace
}  // namespace machina
