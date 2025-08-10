/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "machina/lite/delegates/flex/buffer_map.h"

#include <sys/types.h>

#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/delegates/flex/buffer_map_util.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/string_util.h"
#include "machina/lite/testing/util.h"
#include "machina/lite/util.h"

namespace tflite {
namespace flex {
namespace {

using ::testing::ElementsAre;

// A bit of RAII to simplify handling of TfLiteTensors in the tests.
using UniqueTfLiteTensor =
    std::unique_ptr<TfLiteTensor, std::function<void(TfLiteTensor*)>>;

template <typename T>
UniqueTfLiteTensor MakeLiteTensor(const std::vector<int>& shape,
                                  const std::vector<T>& data) {
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(T), tensor.get());
  memcpy(tensor->data.raw, data.data(), data.size() * sizeof(T));
  return tensor;
}

template <>
UniqueTfLiteTensor MakeLiteTensor<string>(const std::vector<int>& shape,
                                          const std::vector<string>& data) {
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<string>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(string), tensor.get());

  DynamicBuffer b;
  for (const string& s : data) {
    b.AddString(s.data(), s.size());
  }
  b.WriteToTensor(tensor.get(), ConvertVectorToTfLiteIntArray(shape));
  return tensor;
}

template <typename T>
machina::Tensor MakeTensor(const std::vector<int64_t>& shape,
                              const std::vector<T>& data,
                              machina::DataType dtype) {
  machina::Tensor tensor(dtype, machina::TensorShape(shape));
  memcpy(tensor.data(), data.data(), data.size() * sizeof(T));
  return tensor;
}

std::vector<int64_t> GetTensorShape(const machina::Tensor& t) {
  std::vector<int64_t> shape(t.dims());
  for (int i = 0; i < t.dims(); ++i) {
    shape[i] = t.dim_size(i);
  }
  return shape;
}

template <typename T>
std::vector<T> GetTensorData(const machina::Tensor& t) {
  const T* data = t.flat<T>().data();
  return std::vector<T>(data, data + t.NumElements());
}

TEST(BufferMapTest, EmptyBuffer) {
  BufferMap buffer_map;
  EXPECT_FALSE(buffer_map.HasTensor(0));
}

TEST(BufferMapTest, SetFromTfLite) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  machina::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), machina::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteString) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<string>({1, 2, 1, 3}, {"", "", "", "str1", "", ""});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<machina::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "str1", "", ""));

  // Also check details of the tensor.
  machina::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), machina::DT_STRING);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, SetFromTfLiteStringTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<string>({1, 2, 4}, {"", "", "", "s3", "", "", "s1", "s2"});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<machina::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "s3", "", "", "s1", "s2"));
}

TEST(BufferMapTest, SetFromTfLiteBuiltinResource) {
  BufferMap buffer_map;

  // Constructs a fake resource tensor.
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = kTfLiteResource;
  tensor->dims = ConvertVectorToTfLiteIntArray({1});
  TfLiteTensorRealloc(sizeof(int32_t), tensor.get());
  tensor->delegate = nullptr;
  tensor->data.i32[0] = 1;

  buffer_map.SetFromTfLite(0, tensor.get());
  // Also check details of the tensor.
  machina::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), machina::DT_RESOURCE);
  ASSERT_EQ(out_tensor.NumElements(), 1);
  machina::ResourceHandle handle =
      out_tensor.flat<machina::ResourceHandle>()(0);
  EXPECT_EQ(handle.name(), "tflite_resource_variable:1");
}

TEST(BufferMapTest, SetFromTensorFlow) {
  machina::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, machina::DT_FLOAT);

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  machina::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), machina::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTensorFlowTwice) {
  machina::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, machina::DT_FLOAT);
  machina::Tensor t2 = MakeTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2},
                                          machina::DT_INT32);
  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTensorFlow(0, t2);

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TfLiteOverwritesTensorFlow) {
  machina::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, machina::DT_FLOAT);
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TensorFlowOverwritesTfLite) {
  machina::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, machina::DT_FLOAT);
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t2.get());
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));
}

TEST(BufferMapTest, TensorflowBufferReuse) {
  const int kAllocationSize = 1000;
  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.data.raw = nullptr;
  TfLiteTensorRealloc(kAllocationSize, &tensor);
  CHECK(tensor.data.raw);
  EXPECT_EQ(tensor.bytes, kAllocationSize);

  TfLiteTensorBuffer* tensor_buffer_reused = new TfLiteTensorBuffer(&tensor);
  // Checks that the underlying buffer is reused.
  EXPECT_TRUE(tensor_buffer_reused->BufferReusedFromTfLiteTensor());
  EXPECT_EQ(tensor_buffer_reused->data(), tensor.data.raw);
  tensor_buffer_reused->Unref();

  TfLiteTensorDataFree(&tensor);
}

TEST(BufferMapTest, ExplicitlyDisableBufferReuse) {
  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.data.raw = nullptr;
  TfLiteTensorRealloc(10, &tensor);
  CHECK(tensor.data.raw);
  EXPECT_EQ(tensor.bytes, 10);

  TfLiteTensorBuffer* tensor_buffer =
      new TfLiteTensorBuffer(&tensor, /*=allow_reusing*/ false);
  // Checks that the underlying buffer is not reused.
  EXPECT_FALSE(tensor_buffer->BufferReusedFromTfLiteTensor());
  EXPECT_NE(tensor_buffer->data(), tensor.data.raw);
  tensor_buffer->Unref();

  TfLiteTensorDataFree(&tensor);
}

}  // namespace
}  // namespace flex
}  // namespace tflite
