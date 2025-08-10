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

#include "machina/lite/delegates/gpu/common/model_builder_helper.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/core/c/common.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(ModelBuilderHelperTest, CreateVectorCopyDataDifferentSize) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteInt32;
  int32_t src_data[4] = {1, 2, 3, 4};
  tflite_tensor.data.i32 = src_data;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = sizeof(src_data) / sizeof(src_data[0]);
  tflite_tensor.bytes = sizeof(src_data);

  int16_t dst[4];
  ASSERT_OK(CreateVectorCopyData(tflite_tensor, dst));
  EXPECT_THAT(dst, ElementsAre(1, 2, 3, 4));

  TfLiteIntArrayFree(tflite_tensor.dims);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
