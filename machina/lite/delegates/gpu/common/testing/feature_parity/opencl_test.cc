/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/common/testing/feature_parity/feature_parity.h"
#include "machina/lite/delegates/gpu/common/testing/feature_parity/utils.h"
#include "machina/lite/delegates/gpu/delegate.h"
#include "machina/lite/delegates/gpu/delegate_options.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {

class OpenCLBackend : public testing::TestWithParam<TestParams> {};

TEST_P(OpenCLBackend, DelegateTest) {
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
      cl_delegate(TfLiteGpuDelegateV2Create(&options),
                  TfLiteGpuDelegateV2Delete);
  const TestParams& param = GetParam();
  const std::vector<uint8_t>& model_binary = param.model;
  ASSERT_NE(model_binary.empty(), true);
  std::unique_ptr<Interpreter> tflite_cpu_interpreter;
  ASSERT_OK(
      BuildInterpreter(GetModel(model_binary.data()), &tflite_cpu_interpreter));
  ASSERT_OK(AllocateTensors(&tflite_cpu_interpreter));
  std::unique_ptr<Interpreter> opencl_interpreter;
  ASSERT_OK(
      BuildInterpreter(GetModel(model_binary.data()), &opencl_interpreter));
  // Ensures that tensors are allocated.
  ASSERT_OK(ModifyGraphWithDelegate(&opencl_interpreter, cl_delegate.get()));
  // Inputs are initialized with consequent values of the fixed range.
  InitializeInputs(/*left=*/0, /*right=*/100, &tflite_cpu_interpreter);
  InitializeInputs(/*left=*/0, /*right=*/100, &opencl_interpreter);
  ASSERT_OK(Invoke(&tflite_cpu_interpreter));
  ASSERT_OK(Invoke(&opencl_interpreter));
  for (int i = 0; i < tflite_cpu_interpreter->outputs().size(); ++i) {
    int id = tflite_cpu_interpreter->outputs()[i];
    const TfLiteTensor* cpu = tflite_cpu_interpreter->tensor(id);
    const TfLiteTensor* gpu = opencl_interpreter->tensor(id);
    EXPECT_THAT(*gpu, TensorEq(testing::FloatNear(1e-6), *cpu))
        << " for output tensor #" << i << " with id " << id;
  }
}

INSTANTIATE_TEST_SUITE_P(FeatureParityTests, OpenCLBackend,
                         testing::ValuesIn(GetFeatureParity()),
                         [](const testing::TestParamInfo<TestParams>& info) {
                           return info.param.name;
                         });
}  // namespace tflite
