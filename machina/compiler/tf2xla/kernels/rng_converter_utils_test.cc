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

#include "machina/compiler/tf2xla/kernels/rng_converter_utils.h"

#include <gtest/gtest.h>
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/rng_alg.h"

namespace machina {
namespace {

TEST(RngConverterUtilsTest, DefaultRngForCPUEqualsGPU) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(DEVICE_CPU_MACHINA_XLAJIT),
            DefaultRngAlgForDeviceType(DEVICE_GPU_MACHINA_XLAJIT));
}

TEST(RngConverterUtilsTest, UnknownDeviceIsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(/*device_type_string=*/"UNKNOWN DEVICE"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, TensorflowAutoSelects) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_DEFAULT),
            machina::RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, ToTensorflow) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_PHILOX),
            machina::RNG_ALG_PHILOX);

  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_THREE_FRY),
            machina::RNG_ALG_THREEFRY);
}

}  // namespace
}  // namespace machina
