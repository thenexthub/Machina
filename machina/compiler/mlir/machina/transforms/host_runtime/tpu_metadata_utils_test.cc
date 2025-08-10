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
#include "machina/compiler/mlir/machina/transforms/host_runtime/tpu_metadata_utils.h"

#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "tsl/platform/protobuf.h"

namespace mlir {
namespace TFTPU {
namespace {

using mlir::DialectRegistry;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;

// TODO(b/229726259): Make EqualsProto available in OSS
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p, testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

std::string TestDataPath() {
  return machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/machina/transforms/host_runtime/testdata/");
}

class TpuMetadataUtilsTest : public ::testing::Test {
 public:
  TpuMetadataUtilsTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  absl::StatusOr<std::vector<mlir::tf_device::ClusterFuncOp>> GetClusterFuncOps(
      absl::string_view mlir_module_filename) {
    TF_RETURN_IF_ERROR(CreateMlirModule(mlir_module_filename));
    std::vector<mlir::tf_device::ClusterFuncOp> cluster_func_ops;

    mlir_module_->walk([&](mlir::tf_device::ClusterFuncOp op) {
      cluster_func_ops.push_back(op);
    });
    return cluster_func_ops;
  }

 private:
  absl::Status CreateMlirModule(absl::string_view mlir_module_filename) {
    std::string mlir_module_path =
        absl::StrCat(TestDataPath(), mlir_module_filename);
    mlir_module_ =
        mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
    if (!mlir_module_) {
      return absl::Status(
          absl::StatusCode::kNotFound,
          absl::StrCat("Could not find MLIR module at ", mlir_module_path));
    }
    return absl::OkStatus();
  }

  DialectRegistry registry_;
  MLIRContext context_;
  OwningOpRef<mlir::ModuleOp> mlir_module_;
};

TEST_F(TpuMetadataUtilsTest, SingleDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto cluster_func_ops,
                          GetClusterFuncOps("basic_cluster.mlir"));
  mlir::tf_device::ClusterFuncOp cluster_func_op = cluster_func_ops.front();

  machina::tpu::TPUCompileMetadataProto compile_metadata;

  ASSERT_TRUE(mlir::succeeded(SetMetadataProtoFromClusterFuncOp(
      cluster_func_op,
      /*num_replicas=*/1, /*num_cores_per_replica=*/1, {}, &compile_metadata)));

  machina::tpu::TPUCompileMetadataProto expected_compile_metadata;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        num_replicas: 1 num_cores_per_replica: 1
      )pb",
      &expected_compile_metadata));

  EXPECT_THAT(compile_metadata, EqualsProto(expected_compile_metadata));
}

TEST_F(TpuMetadataUtilsTest, spmd) {
  TF_ASSERT_OK_AND_ASSIGN(auto cluster_func_ops,
                          GetClusterFuncOps("spmd.mlir"));
  mlir::tf_device::ClusterFuncOp cluster_func_op = cluster_func_ops.front();

  machina::tpu::TPUCompileMetadataProto compile_metadata;

  ASSERT_TRUE(mlir::succeeded(SetMetadataProtoFromClusterFuncOp(
      cluster_func_op,
      /*num_replicas=*/1, /*num_cores_per_replica=*/2, {}, &compile_metadata)));

  machina::tpu::TPUCompileMetadataProto expected_compile_metadata;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        args {
          dtype: DT_FLOAT
          shape { unknown_rank: true }
          kind: PARAMETER
          sharding {
            type: OTHER
            tile_assignment_dimensions: 2
            tile_assignment_dimensions: 1
            tile_assignment_devices: 0
            tile_assignment_devices: 1
          }
          is_bounded_dynamic_dim: false
        }
        retvals { sharding {} }
        num_replicas: 1
        num_cores_per_replica: 2
        use_spmd_for_xla_partitioning: true
        compile_options {}
      )pb",
      &expected_compile_metadata));

  EXPECT_THAT(compile_metadata, EqualsProto(expected_compile_metadata));
}

}  // namespace
}  // namespace TFTPU
}  // namespace mlir
