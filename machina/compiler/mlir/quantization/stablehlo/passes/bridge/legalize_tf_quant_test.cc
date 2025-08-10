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

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tf2xla/api/v2/legalize_tf.h"
#include "machina/xla/client/client_library.h"
#include "machina/xla/shape.h"
#include "machina/xla/stream_executor/platform.h"
#include "machina/xla/stream_executor/platform_manager.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"

namespace mlir::quant::stablehlo {
namespace {

using ::testing::Test;

class LegalizeTFQuantTest : public Test {
 protected:
  void TestBridgeLowering(toolchain::StringRef mlir_module_string,
                          toolchain::ArrayRef<machina::TensorShape> arg_shapes,
                          machina::DataType dtype) {
    machina::tpu::MlirToHloArgs mlir_to_hlo_args;
    mlir_to_hlo_args.rollout_state =
        machina::ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
    mlir_to_hlo_args.mlir_module = mlir_module_string;
    machina::se::Platform* platform =
        machina::se::PlatformManager::PlatformWithName("Host").value();
    auto client =
        xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform).value();
    machina::tpu::TPUCompileMetadataProto metadata_proto;
    // Set up an arg per arg_shape with the specified type.
    for (int i = 0; i < arg_shapes.size(); ++i) {
      auto metadata_arg = metadata_proto.add_args();
      metadata_arg->set_kind(
          machina::tpu::TPUCompileMetadataProto::Arg::PARAMETER);
      metadata_arg->set_dtype(dtype);
    }
    // Set up one dummy retval.
    metadata_proto.add_retvals();
    bool use_tuple_args = true;
    std::vector<machina::tpu::ShardingAndIndex> arg_core_mapping;
    std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

    TF_EXPECT_OK(machina::tf2xla::v2::LegalizeMlirToHlo(
                     mlir_to_hlo_args, metadata_proto, use_tuple_args,
                     /*device_type=*/"MACHINA_MACHINA_XLA_TPU_JIT", custom_legalization_passes,
                     /*shape_determination_fns=*/{}, arg_shapes,
                     &arg_core_mapping, &per_core_arg_shapes, client)
                     .status());
  }
};

TEST_F(LegalizeTFQuantTest, LegalizesModuleWithTFUniformQuantization) {
  constexpr char mlir_module_string[] = R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>
      %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
      func.return %1 : tensor<1xf32>
    }
  })mlir";

  std::vector<machina::TensorShape> arg_shapes = {{1}};

  TestBridgeLowering(mlir_module_string, arg_shapes, machina::DT_FLOAT);
}

TEST_F(LegalizeTFQuantTest, LegalizesModuleWithDequantize) {
  constexpr char mlir_module_string[] = R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xf32> {
      %min_range = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %max_range = "tf.Const"() { value = dense<5.0> : tensor<f32> } : () -> tensor<f32>
      %0 = "tf.Dequantize"(%arg0, %min_range, %max_range) : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> tensor<1xf32>
      func.return %0 : tensor<1xf32>
    }
  })mlir";
  std::vector<machina::TensorShape> arg_shapes = {{1}};

  TestBridgeLowering(mlir_module_string, arg_shapes, machina::DT_QINT8);
}

TEST_F(LegalizeTFQuantTest, LegalizesModuleWithClipByValue) {
  constexpr char mlir_module_string[] = R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
      %max = "tf.Const"() { value = dense<12.0> : tensor<f32> } : () -> tensor<f32>
      %min = "tf.Const"() { value = dense<-25.0> : tensor<f32> } : () -> tensor<f32>
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint32>
      %qmax = "tf.UniformQuantize"(%max, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<!tf_type.qint32>
      %qmin = "tf.UniformQuantize"(%min, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<!tf_type.qint32>

      %1 = "tf.UniformQuantizedClipByValue"(%0, %qmin, %qmax, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint32>

      %2 = "tf.UniformDequantize"(%1, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
      } : (tensor<2x2x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<2x2xf32>
      func.return %2 : tensor<2x2xf32>
    }
  })mlir";
  std::vector<machina::TensorShape> arg_shapes = {{2, 2}};

  TestBridgeLowering(mlir_module_string, arg_shapes, machina::DT_FLOAT);
}

}  // namespace
}  // namespace mlir::quant::stablehlo
