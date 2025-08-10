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

class ListLengthModel : public ListOpModel {
 public:
  ListLengthModel() {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    length_output_ = AddOutput({TensorType_INT32, {}});
    SetCustomOp("ListLength", {}, Register_LIST_LENGTH);
    BuildInterpreter({{}});
  }
  const TfLiteTensor* GetOutputTensor() {
    return interpreter_->tensor(length_output_);
  }
  int list_input_;
  int length_output_;
};

class ListLengthTest : public ::testing::TestWithParam<int> {};

TEST_P(ListLengthTest, OutputIsListLength) {
  const int length = GetParam();
  ListLengthModel m;
  m.PopulateListTensor(m.list_input_, {2, 2}, length, kTfLiteInt32);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const output = m.GetOutputTensor();
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(output->allocation_type, kTfLiteArenaRw);
  ASSERT_THAT(output, DimsAre({}));
  ASSERT_EQ(output->data.i32[0], length);
}

INSTANTIATE_TEST_SUITE_P(ListLengthTests, ListLengthTest,
                         testing::Values(0, 1, 5, 10, 100));

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
