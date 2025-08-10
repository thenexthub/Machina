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
#ifndef MACHINA_LITE_KERNELS_ADD_N_TEST_COMMON_H_
#define MACHINA_LITE_KERNELS_ADD_N_TEST_COMMON_H_

#include <vector>

#include "machina/lite/kernels/test_util.h"

namespace tflite {

class BaseAddNOpModel : public SingleOpModel {
 public:
  BaseAddNOpModel(const std::vector<TensorData>& inputs,
                  const TensorData& output) {
    int num_inputs = inputs.size();
    std::vector<std::vector<int>> input_shapes;

    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(AddInput(inputs[i]));
      input_shapes.push_back(GetShape(inputs_[i]));
    }

    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD_N, BuiltinOptions_AddNOptions,
                 CreateAddNOptions(builder_).Union());
    BuildInterpreter(input_shapes);
  }

  int input(int i) { return inputs_[i]; }

 protected:
  std::vector<int> inputs_;
  int output_;
};

class FloatAddNOpModel : public BaseAddNOpModel {
 public:
  using BaseAddNOpModel::BaseAddNOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerAddNOpModel : public BaseAddNOpModel {
 public:
  using BaseAddNOpModel::BaseAddNOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_ADD_N_TEST_COMMON_H_
