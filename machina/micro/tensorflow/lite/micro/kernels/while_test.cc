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
#include "machina/lite/micro/micro_arena_constants.h"
#include "machina/lite/micro/micro_interpreter.h"
#include "machina/lite/micro/mock_micro_graph.h"
#include "machina/lite/micro/test_helper_custom_ops.h"
#include "machina/lite/micro/test_helpers.h"
#include "machina/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(WhileShouldNeverInvokeConditionFalse) {
  constexpr int kArenaSize = 5000;
  uint8_t arena[kArenaSize];

  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithSubgraphsAndWhile();
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddWhile();
  resolver.AddAdd();
  resolver.AddLess();
  tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* input0 = interpreter.input(0);
  TfLiteTensor* input1 = interpreter.input(1);
  TfLiteTensor* output0 = interpreter.output(0);
  TfLiteTensor* output1 = interpreter.output(1);
  input0->data.f[0] = 3.0f;
  input1->data.f[0] = 2.0f;

  interpreter.Invoke();

  TF_LITE_MICRO_EXPECT_EQ(output0->data.f[0], 3.0f);
  TF_LITE_MICRO_EXPECT_EQ(output1->data.f[0], 2.0f);
}

TF_LITE_MICRO_TEST(WhileShouldInvokeOnce) {
  constexpr int kArenaSize = 5000;
  uint8_t arena[kArenaSize];

  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithSubgraphsAndWhile();
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddWhile();
  resolver.AddAdd();
  resolver.AddLess();
  tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* input0 = interpreter.input(0);
  TfLiteTensor* input1 = interpreter.input(1);
  TfLiteTensor* output0 = interpreter.output(0);
  TfLiteTensor* output1 = interpreter.output(1);
  input0->data.f[0] = 2.0f;
  input1->data.f[0] = 3.0f;

  interpreter.Invoke();

  TF_LITE_MICRO_EXPECT_EQ(output0->data.f[0], 5.0f);
  TF_LITE_MICRO_EXPECT_EQ(output1->data.f[0], 3.0f);
}

TF_LITE_MICRO_TESTS_END
