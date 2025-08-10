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
#include "machina/lite/experimental/ml_adjacent/tflite/tfl_tensor_ref.h"

#include <algorithm>
#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/experimental/ml_adjacent/lib.h"
#include "machina/lite/kernels/internal/compatibility.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/util.h"

namespace ml_adj {
namespace data {
namespace {

using ::testing::Each;
using ::tflite::BuildTfLiteTensor;
using ::tflite::DimsAre;
using ::tflite::NumElements;
using ::tflite::TensorUniquePtr;

// Mock implementation of `ResizeTensor`.
TfLiteStatus SimpleResizeTensor(TfLiteContext*, TfLiteTensor* tensor,
                                TfLiteIntArray* new_size) {
  TFLITE_CHECK(tensor->type == kTfLiteFloat32);
  size_t num_bytes = NumElements(new_size) * sizeof(float);
  TF_LITE_ENSURE_STATUS(TfLiteTensorRealloc(num_bytes, tensor));
  if (tensor->dims != nullptr) {
    TfLiteIntArrayFree(tensor->dims);
  }
  tensor->dims = new_size;
  return kTfLiteOk;
}

std::unique_ptr<TfLiteContext> MakeSimpleContext() {
  auto ctx = std::make_unique<TfLiteContext>();
  ctx->ResizeTensor = SimpleResizeTensor;
  return ctx;
}

TEST(ImmutableTensorRefTest, ConstructsAndManifestsTensorData) {
  TensorUniquePtr tfl_tensor =
      BuildTfLiteTensor(kTfLiteFloat32, {2, 2}, kTfLiteDynamic);
  std::fill(tfl_tensor->data.f, tfl_tensor->data.f + 4, 2.0f);
  TflTensorRef ref(tfl_tensor.get());

  ASSERT_EQ(ref.Type(), etype_t::f32);
  ASSERT_EQ(ref.Dims(), (dims_t{2, 2}));
  ASSERT_EQ(ref.Bytes(), 4 * sizeof(float));

  absl::Span<const float> data(reinterpret_cast<const float*>(ref.Data()), 4);
  EXPECT_THAT(data, Each(2.0f));
}

TEST(MutableTensorRefTest, ConstructsAndManifestsTensorData) {
  TensorUniquePtr tfl_tensor =
      BuildTfLiteTensor(kTfLiteFloat32, {2, 2}, kTfLiteDynamic);
  std::fill(tfl_tensor->data.f, tfl_tensor->data.f + 4, 2.0f);
  MutableTflTensorRef ref(tfl_tensor.get(), nullptr);

  ASSERT_EQ(ref.Type(), etype_t::f32);
  ASSERT_EQ(ref.Dims(), (dims_t{2, 2}));
  ASSERT_EQ(ref.Bytes(), 4 * sizeof(float));

  absl::Span<const float> data(reinterpret_cast<const float*>(ref.Data()), 4);
  EXPECT_THAT(data, Each(2.0f));
}

TEST(MutableTensorRefTest, TensorRefWritesDataToTensor) {
  TensorUniquePtr tfl_tensor =
      BuildTfLiteTensor(kTfLiteFloat32, {3, 3}, kTfLiteDynamic);
  MutableTflTensorRef ref(tfl_tensor.get(), nullptr);

  ASSERT_EQ(ref.Type(), etype_t::f32);
  ASSERT_EQ(ref.Dims(), (dims_t{3, 3}));
  ASSERT_EQ(ref.Bytes(), 9 * sizeof(float));

  absl::Span<float> data(reinterpret_cast<float*>(ref.Data()), 9);
  std::fill(data.begin(), data.end(), 3.0f);

  EXPECT_THAT(absl::Span<const float>(tfl_tensor->data.f, 9), Each(3.0f));
}

TEST(MutableTensorRefTest, ResizeIncreaseSize) {
  TensorUniquePtr tfl_tensor =
      BuildTfLiteTensor(kTfLiteFloat32, {2, 3}, kTfLiteDynamic);
  std::unique_ptr<TfLiteContext> ctx = MakeSimpleContext();
  MutableTflTensorRef ref(tfl_tensor.get(), ctx.get());

  ASSERT_EQ(ref.Type(), etype_t::f32);
  ASSERT_EQ(ref.Dims(), (dims_t{2, 3}));
  ASSERT_EQ(ref.Bytes(), 6 * sizeof(float));

  ref.Resize({3, 3});
  ASSERT_EQ(ref.Dims(), (dims_t{3, 3}));
  ASSERT_EQ(ref.Bytes(), 9 * sizeof(float));

  // Sanitizers will check buffer is correct size.
  absl::Span<float> ref_data(reinterpret_cast<float*>(ref.Data()), 9);

  // Check underlying tensor is also resized.
  ASSERT_THAT(tfl_tensor.get(), DimsAre({3, 3}));
  ASSERT_EQ(tfl_tensor->bytes, ref.Bytes());

  // Check share same buffer.
  ASSERT_EQ(ref.Data(), tfl_tensor->data.data);
}

TEST(MutableTensorRefTest, ResizeDecreasesSize) {
  TensorUniquePtr tfl_tensor =
      BuildTfLiteTensor(kTfLiteFloat32, {2, 3}, kTfLiteDynamic);
  std::unique_ptr<TfLiteContext> ctx = MakeSimpleContext();
  MutableTflTensorRef ref(tfl_tensor.get(), ctx.get());

  ASSERT_EQ(ref.Type(), etype_t::f32);
  ASSERT_EQ(ref.Dims(), (dims_t{2, 3}));
  ASSERT_EQ(ref.Bytes(), 6 * sizeof(float));

  ref.Resize({2, 2});
  ASSERT_EQ(ref.Dims(), (dims_t{2, 2}));
  ASSERT_EQ(ref.Bytes(), 4 * sizeof(float));

  // Sanitizers will check buffer is correct size.
  absl::Span<float> ref_data(reinterpret_cast<float*>(ref.Data()), 4);

  // Check underlying tensor is also resized.
  ASSERT_THAT(tfl_tensor.get(), DimsAre({2, 2}));
  ASSERT_EQ(tfl_tensor->bytes, ref.Bytes());

  // Check share same buffer.
  ASSERT_EQ(ref.Data(), tfl_tensor->data.data);
}

}  // namespace
}  // namespace data
}  // namespace ml_adj
