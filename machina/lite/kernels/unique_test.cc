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
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T, typename I>
class UniqueOpModel : public SingleOpModel {
 public:
  UniqueOpModel(const TensorData& input, TensorType input_type,
                TensorType index_out_type) {
    input_id_ = AddInput(input);
    output_id_ = AddOutput(input_type);
    output_index_id_ = AddOutput(index_out_type);
    SetBuiltinOp(BuiltinOperator_UNIQUE, BuiltinOptions_UniqueOptions,
                 CreateUniqueOptions(builder_, index_out_type).Union());
    BuildInterpreter({GetShape(input_id_)});
  }

  int input_tensor_id() { return input_id_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<I> GetIndexesOutput() {
    return ExtractVector<I>(output_index_id_);
  }

 protected:
  int input_id_;
  int output_id_;
  int output_index_id_;
};

TEST(UniqueOpModelTest, OneElement) {
  UniqueOpModel<float, int32_t> model({TensorType_FLOAT32, {1}},
                                      TensorType_FLOAT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input_tensor_id(), {5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5}));
  EXPECT_THAT(model.GetIndexesOutput(), ElementsAreArray({0}));
}

TEST(UniqueOpModelTest, MultipleElements_AllUnique) {
  UniqueOpModel<float, int32_t> model({TensorType_FLOAT32, {8}},
                                      TensorType_FLOAT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input_tensor_id(),
                              {5, 2, 3, 51, 6, 72, 7, 8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5, 2, 3, 51, 6, 72, 7, 8}));
  EXPECT_THAT(model.GetIndexesOutput(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(UniqueOpModelTest, MultipleElements_AllDuplicates) {
  UniqueOpModel<float, int32_t> model({TensorType_FLOAT32, {7}},
                                      TensorType_FLOAT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input_tensor_id(), {5, 5, 5, 5, 5, 5, 5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5}));
  EXPECT_THAT(model.GetIndexesOutput(),
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0}));
}

TEST(UniqueOpModelTest, MultipleElements_SomeDuplicates) {
  UniqueOpModel<float, int32_t> model({TensorType_FLOAT32, {7}},
                                      TensorType_FLOAT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input_tensor_id(), {2, 3, 5, 7, 2, 7, 3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 3, 5, 7}));
  EXPECT_THAT(model.GetIndexesOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 3, 1}));
}

TEST(UniqueOpModelTest, MultipleElements_RepeatedDuplicates) {
  UniqueOpModel<float, int32_t> model({TensorType_FLOAT32, {6}},
                                      TensorType_FLOAT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input_tensor_id(),
                              {-1, -1, -2, -2, -3, -3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({-1, -2, -3}));
  EXPECT_THAT(model.GetIndexesOutput(), ElementsAreArray({0, 0, 1, 1, 2, 2}));
}

TEST(UniqueOpModelTest, MultipleElements_SomeDuplicates_IndexInt64) {
  UniqueOpModel<float, int64_t> model({TensorType_FLOAT32, {7}},
                                      TensorType_FLOAT32, TensorType_INT64);
  model.PopulateTensor<float>(model.input_tensor_id(), {2, 3, 5, 7, 2, 7, 3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 3, 5, 7}));
  EXPECT_THAT(model.GetIndexesOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 3, 1}));
}

}  // namespace
}  // namespace tflite
