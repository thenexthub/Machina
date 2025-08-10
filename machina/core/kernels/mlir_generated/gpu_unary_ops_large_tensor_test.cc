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

#include <cmath>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "machina/core/framework/device.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/mlir_generated/base_ops_test.h"
#include "machina/core/kernels/mlir_generated/base_unary_ops_test.h"

namespace machina {
namespace {

// Test fixture `UnaryOpsLargeTensorTest` that sets the TF device is expected by
// the TEST macros below.
class UnaryOpsLargeTensorTest : public UnaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<machina::Device> device_gpu(
        machina::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(machina::DEVICE_GPU, std::move(device_gpu));
  }
};

/// Test `tf.Abs`.

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TEST_F(UnaryOpsLargeTensorTest, Abs) {
  Test<float, float, float, float>(
      "Abs", test::DefaultInputShapeExceedingInt32(),
      test::DefaultInput<float>(), std::abs,
      test::OpsTestConfig().ExpectStrictlyEqual().SuppressTolerance());
}
#endif

/// Test `tf.Atanh`.

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TEST_F(UnaryOpsLargeTensorTest, Atanh) {
  Test<float, float, float, float>("Atanh",
                                   test::DefaultInputShapeExceedingInt32(),
                                   test::DefaultInput<float>(), std::atanh,
                                   test::OpsTestConfig().ExpectStrictlyEqual());
}
#endif

}  // namespace
}  // namespace machina
