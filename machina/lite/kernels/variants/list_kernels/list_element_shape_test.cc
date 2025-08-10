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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/kernels/variants/list_kernels/test_util.h"
#include "machina/lite/kernels/variants/list_ops_lib.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::testing::ElementsAreArray;

class ListElementShapeModel : public ListOpModel {
 public:
  ListElementShapeModel() {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    shape_output_ = AddOutput({TensorType_INT32, {}});
    SetCustomOp("ListElementShape", {}, Register_LIST_ELEMENT_SHAPE);
    BuildInterpreter({{}});
  }
  const TfLiteTensor* GetOutputTensor(int index) {
    return interpreter_->tensor(index);
  }
  int list_input_;
  int shape_output_;
};

TEST(ListElementShapeTest, MultiDimStaticShape) {
  ListElementShapeModel m;
  m.PopulateListTensor(0, {2, 2}, 10, kTfLiteInt32);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const out = m.GetOutputTensor(m.shape_output_);
  ASSERT_THAT(out, DimsAre({2}));
  ASSERT_THAT(std::vector<int>(out->data.i32, out->data.i32 + 2),
              ElementsAreArray({2, 2}));
}

TEST(ListElementShapeTest, MultiDimWithDynamicDims) {
  ListElementShapeModel m;
  m.PopulateListTensor(0, {2, -1, 3}, 10, kTfLiteInt32);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const out = m.GetOutputTensor(m.shape_output_);
  ASSERT_THAT(out, DimsAre({3}));
  ASSERT_THAT(std::vector<int>(out->data.i32, out->data.i32 + 3),
              ElementsAreArray({2, -1, 3}));
}

TEST(ListElementShapeTest, ScalarShape) {
  ListElementShapeModel m;
  m.PopulateListTensor(0, {0}, 10, kTfLiteInt32);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const out = m.GetOutputTensor(m.shape_output_);
  ASSERT_THAT(out, DimsAre({0}));
  ASSERT_EQ(out->bytes, 0);
}

TEST(ListElementShapeTest, UnrankedShape) {
  ListElementShapeModel m;
  m.PopulateListTensor(0, {}, 10, kTfLiteInt32);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const out = m.GetOutputTensor(m.shape_output_);
  ASSERT_THAT(out, DimsAre({}));
  ASSERT_EQ(out->bytes, sizeof(int));
  ASSERT_EQ(out->data.i32[0], -1);
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
