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
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "machina/lite/kernels/add_n_test_common.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(FloatAddNOpModel, AddMultipleTensors) {
  FloatAddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}}},
                     {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  m.PopulateTensor<float>(m.input(2), {0.5, 0.1, 0.1, 0.2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              Pointwise(FloatingPointEq(), {-1.4, 0.5, 1.1, 1.5}));
}

TEST(FloatAddNOpModel, Add2Tensors) {
  FloatAddNOpModel m(
      {{TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}}},
      {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              Pointwise(FloatingPointEq(), {-1.9, 0.4, 1.0, 1.3}));
}

TEST(IntegerAddNOpModel, AddMultipleTensors) {
  IntegerAddNOpModel m({{TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input(0), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input(1), {1, 2, 3, 5});
  m.PopulateTensor<int32_t>(m.input(2), {10, -5, 1, -2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-9, -1, 11, 11}));
}

}  // namespace
}  // namespace tflite
