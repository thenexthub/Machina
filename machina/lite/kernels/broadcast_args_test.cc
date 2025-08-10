/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "machina/lite/core/interpreter.h"
#include "machina/lite/core/kernels/register.h"
#include "machina/lite/core/model.h"
#include "machina/lite/kernels/test_util.h"

namespace tflite {
namespace {
using ::testing::ElementsAreArray;

template <class ShapeType = int32_t>
class BroadcastArgsOpModel : public SingleOpModel {
 public:
  BroadcastArgsOpModel(std::initializer_list<ShapeType> input1,
                       std::initializer_list<ShapeType> input2,
                       bool constant_tensor) {
    int input1_length = input1.size();
    int input2_length = input2.size();
    if (constant_tensor) {
      shape1_ =
          AddConstInput({GetTensorType<ShapeType>(), {input1_length}}, input1);
      shape2_ =
          AddConstInput({GetTensorType<ShapeType>(), {input2_length}}, input2);
    } else {
      shape1_ = AddInput({GetTensorType<ShapeType>(), {input1_length}});
      shape2_ = AddInput({GetTensorType<ShapeType>(), {input2_length}});
    }
    output_ = AddOutput(GetTensorType<ShapeType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_ARGS, BuiltinOptions_NONE, 0);
    BuildInterpreter({{input1_length}, {input2_length}});
    if (!constant_tensor) {
      if (input1.size() > 0) SetInput1(input1);
      if (input2.size() > 0) SetInput2(input2);
    }
  }

  void SetInput1(std::initializer_list<ShapeType> data) {
    PopulateTensor(shape1_, data);
  }

  void SetInput2(std::initializer_list<ShapeType> data) {
    PopulateTensor(shape2_, data);
  }

  std::vector<ShapeType> GetOutput() {
    return ExtractVector<ShapeType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int shape1_;
  int shape2_;
  int output_;
};

template <typename T>
class BroadcastArgsOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<int64_t, int32_t>;
TYPED_TEST_SUITE(BroadcastArgsOpTest, DataTypes);

#if GTEST_HAS_DEATH_TEST
TYPED_TEST(BroadcastArgsOpTest, ShapeNotBroadcastableConstant) {
  EXPECT_DEATH(BroadcastArgsOpModel<TypeParam> m({2, 3, 4, 4}, {2, 2},
                                                 /*constant_tensor=*/true),
               "");
}

TYPED_TEST(BroadcastArgsOpTest, ShapeNotBroadcastable) {
  BroadcastArgsOpModel<TypeParam> m({2, 3, 4, 4}, {2, 2},
                                    /*constant_tensor=*/false);
  EXPECT_DEATH(ASSERT_EQ(m.Invoke(), kTfLiteOk), "");
}
#endif

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsWithScalar) {
  for (bool constant_tensor : {true, false}) {
    BroadcastArgsOpModel<TypeParam> m({}, {2, 4}, constant_tensor);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 4}));
  }
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsDifferentDims) {
  for (bool constant_tensor : {true, false}) {
    BroadcastArgsOpModel<TypeParam> m({1}, {2, 4}, constant_tensor);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 4}));
  }
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsSameDims) {
  for (bool constant_tensor : {true, false}) {
    BroadcastArgsOpModel<TypeParam> m({1, 4, 6, 3, 1, 5}, {4, 4, 1, 3, 4, 1},
                                      constant_tensor);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 4, 6, 3, 4, 5}));
  }
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsComplex) {
  for (bool constant_tensor : {true, false}) {
    BroadcastArgsOpModel<TypeParam> m({6, 3, 1, 5}, {4, 4, 1, 3, 4, 1},
                                      constant_tensor);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 4, 6, 3, 4, 5}));
  }
}

}  // namespace
}  // namespace tflite
