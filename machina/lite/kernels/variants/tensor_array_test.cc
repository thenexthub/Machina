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

#include "machina/lite/kernels/variants/tensor_array.h"

#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/array.h"
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/c/common.h"
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/portable_type_to_tflitetype.h"
#include "machina/lite/util.h"

namespace tflite {
namespace variants {
namespace {

template <typename T>
TensorUniquePtr MakeTensorWithData(std::vector<int> dims,
                                   const std::vector<T>& data) {
  TensorUniquePtr tensor =
      BuildTfLiteTensor(typeToTfLiteType<T>(), dims, kTfLiteDynamic);
  const int num_elements =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  T* data_start = (T*)tensor->data.data;
  // TODO(b/257472333) Investigate vector alignment and if this
  // can be replaced with memcpy.
  for (int i = 0; i < num_elements; ++i) {
    data_start[i] = data[i];
  }
  // For these tests we want to give `TensorArray`s ownership of their
  // constituent tensors, so we release here.
  return tensor;
}

TensorArray MakeTensorArrayForTest(const std::vector<int>& dims) {
  return TensorArray(kTfLiteInt32, BuildTfLiteArray(dims));
}

TEST(TensorArrayTest, InsertSingleElement) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  ASSERT_TRUE(arr.Set(0, MakeTensorWithData<int>({2}, {3, 4})));
  const TfLiteTensor* added_tensor = arr.At(0);
  ASSERT_TRUE(added_tensor != nullptr);
  ASSERT_THAT(added_tensor, DimsAre({2}));
  EXPECT_EQ(added_tensor->data.i32[0], 3);
  EXPECT_EQ(added_tensor->data.i32[1], 4);
}

TEST(TensorArrayTest, ResizeToZero) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  EXPECT_EQ(arr.NumElements(), 2);
  arr.Resize(0);
  EXPECT_EQ(arr.NumElements(), 0);
}

TEST(TensorArrayTest, InsertOOB) {
  auto arr = MakeTensorArrayForTest({});
  TensorUniquePtr tensor = MakeTensorWithData<int>({2}, {3, 4});
  arr.Resize(1);
  ASSERT_FALSE(arr.Set(-1, std::move(tensor)));
  EXPECT_FALSE(arr.At(0));
}

TEST(TensorArrayTest, InsertMultipleElements) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  EXPECT_TRUE(arr.Set(0, MakeTensorWithData<int>({2}, {3, 4})));
  EXPECT_TRUE(arr.Set(1, MakeTensorWithData<int>({3}, {3, 4, 5})));
  EXPECT_THAT(arr.At(0), DimsAre({2}));
  EXPECT_THAT(arr.At(1), DimsAre({3}));
}

TEST(TensorArrayTest, InsertSameIndexTwiceDeletes) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  EXPECT_TRUE(arr.Set(0, MakeTensorWithData<int>({2}, {3, 2})));
  EXPECT_TRUE(arr.Set(0, MakeTensorWithData<int>({3}, {3, 4, 5})));
  EXPECT_THAT(arr.At(0), DimsAre({3}));
}

TEST(TensorArrayTest, ResizeUpWithElements) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(1);
  ASSERT_TRUE(arr.Set(0, MakeTensorWithData<int>({2}, {3, 4})));
  arr.Resize(2);
  EXPECT_THAT(arr.At(0), DimsAre({2}));
  EXPECT_FALSE(arr.At(1));
  EXPECT_EQ(arr.NumElements(), 2);
}

// resize down delete elements.
TEST(TensorArrayTest, ResizeDownDeletesElements) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  ASSERT_TRUE(arr.Set(1, MakeTensorWithData<int>({2}, {3, 4})));
  arr.Resize(1);
  EXPECT_EQ(arr.NumElements(), 1);
  EXPECT_FALSE(arr.At(0));
}

TEST(TensorArrayTest, CopyListWithZeroLength) {
  auto arr = MakeTensorArrayForTest({});
  TensorArray arr2{arr};
  EXPECT_EQ(arr.NumElements(), arr2.NumElements());
  EXPECT_EQ(arr.NumElements(), 0);
}

TEST(TensorArrayTest, CopyAssignListWithZeroLength) {
  auto arr = MakeTensorArrayForTest({});
  arr = MakeTensorArrayForTest({2, 2});
  EXPECT_EQ(arr.NumElements(), 0);
  EXPECT_THAT(arr.ElementShape(), DimsAre({2, 2}));
}

TEST(TensorArrayTest, CopyEmptyList) {
  auto arr = MakeTensorArrayForTest({});
  arr.Resize(2);
  TensorArray arr2{arr};
  EXPECT_EQ(arr.NumElements(), arr2.NumElements());
  EXPECT_EQ(arr.NumElements(), 2);
}

TEST(TensorArrayTest, CopyAssignToEmptyList) {
  auto arr = MakeTensorArrayForTest({});
  auto target_arr = MakeTensorArrayForTest({2, 2});
  target_arr.Resize(2);
  target_arr = arr;
  EXPECT_EQ(target_arr.NumElements(), 0);
  EXPECT_THAT(target_arr.ElementShape(), DimsAre({}));
}

TEST(TensorArrayTest, CopyListWithItem) {
  std::optional<TensorArray> arr = TensorArray(kTfLiteInt32, {});
  arr->Resize(1);
  ASSERT_TRUE(arr->Set(0, MakeTensorWithData<int>({2}, {3, 4})));

  TensorArray arr2{*arr};
  EXPECT_EQ(arr->NumElements(), arr2.NumElements());
  // Both point to the same tensor.
  EXPECT_EQ(arr->At(0), arr2.At(0));
  // They are ref counted so deleting one list doesn't effect the tensor
  // in the other.
  arr.reset();
  EXPECT_THAT(arr2.At(0), DimsAre({2}));
}

TEST(TensorArrayTest, CopyAssignToListWithItem) {
  auto target_arr = MakeTensorArrayForTest({});
  target_arr.Resize(2);
  ASSERT_TRUE(target_arr.Set(0, MakeTensorWithData<int>({2}, {3, 4})));

  auto src_arr = MakeTensorArrayForTest({2, 2});
  src_arr.Resize(1);

  target_arr = src_arr;

  EXPECT_EQ(target_arr.NumElements(), src_arr.NumElements());
  EXPECT_EQ(target_arr.At(0), nullptr);
}

TEST(TensorArrayTest, CopyAssignFromListWithItem) {
  auto target_arr = MakeTensorArrayForTest({2, 2});
  target_arr.Resize(1);

  auto src_arr = MakeTensorArrayForTest({});
  src_arr.Resize(2);
  ASSERT_TRUE(src_arr.Set(0, MakeTensorWithData<int>({2}, {3, 4})));

  target_arr = src_arr;

  EXPECT_EQ(target_arr.NumElements(), src_arr.NumElements());
  EXPECT_EQ(src_arr.At(0), target_arr.At(0));
}

TEST(TensorArrayTest, DeleteEmptyTensorArray) {
  TensorArray* arr = new TensorArray{kTfLiteInt32, {}};
  delete arr;
}

TEST(TensorArrayTest, DeleteResizedEmptyTensorArray) {
  TensorArray* arr = new TensorArray{kTfLiteInt32, {}};
  arr->Resize(2);
  delete arr;
}

// OpaqueVariantTensorArrayDataTest(s) test usage of the `TensorArray` through
// the generic interface methods defined in
// `third_party/machina/lite/core/c/common.h`. While appearing slightly
// contrived in function, this test exemplifies proper casting protocol of
// `VariantData` and asserts that the derived methods are dispatched to.
TEST(OpaqueVariantTensorArrayDataTest, CastThroughVoidAndCopy) {
  TensorArray* arr = new TensorArray{kTfLiteFloat32, {}};
  arr->Resize(2);
  ASSERT_TRUE(arr->Set(0, MakeTensorWithData<int>({2}, {3, 4})));
  void* erased = static_cast<VariantData*>(arr);

  VariantData* d = static_cast<VariantData*>(erased);
  VariantData* copied_d = d->CloneTo(nullptr);
  auto* copied_arr = static_cast<TensorArray*>(copied_d);
  ASSERT_THAT(copied_arr->At(0), DimsAre({2}));
  ASSERT_THAT(arr->At(0), DimsAre({2}));
  ASSERT_EQ(arr->At(0), arr->At(0));

  delete d;
  delete copied_d;
}

}  // namespace
}  // namespace variants
}  // namespace tflite
