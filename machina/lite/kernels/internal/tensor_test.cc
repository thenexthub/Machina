/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "machina/lite/kernels/internal/tensor.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

using ::testing::ElementsAre;

TEST(TensorTest, GetTensorShape4D) {
  RuntimeShape d = GetTensorShape({2, 3, 4, 5});
  EXPECT_THAT(
      std::vector<int32_t>(d.DimsData(), d.DimsData() + d.DimensionsCount()),
      ElementsAre(2, 3, 4, 5));
}

TEST(TensorTest, GetTensorShape3D) {
  RuntimeShape d = GetTensorShape({3, 4, 5});
  EXPECT_THAT(
      std::vector<int32_t>(d.DimsData(), d.DimsData() + d.DimensionsCount()),
      ElementsAre(3, 4, 5));
}

TEST(TensorTest, GetTensorShape2D) {
  RuntimeShape d = GetTensorShape({4, 5});
  EXPECT_THAT(
      std::vector<int32_t>(d.DimsData(), d.DimsData() + d.DimensionsCount()),
      ElementsAre(4, 5));
}

TEST(TensorTest, GetTensorShape1D) {
  RuntimeShape d = GetTensorShape({5});
  EXPECT_THAT(
      std::vector<int32_t>(d.DimsData(), d.DimsData() + d.DimensionsCount()),
      ElementsAre(5));
}

}  // namespace
}  // namespace tflite
