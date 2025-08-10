/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/micro/kernels/kernel_runner.h"
#include "machina/lite/micro/test_helpers.h"
#include "machina/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestNegFloat(int* input_dims_data, const float* input_data,
                  const float* expected_output_data, int* output_dims_data,
                  float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_NEG();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  TF_LITE_MICRO_EXPECT_EQ(expected_output_data[0], output_data[0]);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(NegOpSingleFloat) {
  int dims[] = {1, 2};
  const float input_data[] = {8.5, 0.0};
  const float golden[] = {-8.5, 0.0};
  float output_data[2];

  tflite::testing::TestNegFloat(dims, input_data, golden, dims, output_data);
}

TF_LITE_MICRO_TEST(NegOpFloat) {
  int dims[] = {2, 2, 3};
  const float input_data[] = {-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f};
  const float golden[] = {2.0f, 1.0f, -0.f, -1.0f, -2.0f, -3.0f};
  float output_data[6];

  tflite::testing::TestNegFloat(dims, input_data, golden, dims, output_data);
}

TF_LITE_MICRO_TESTS_END
