/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TESTING_GENERATE_TESTSPEC_H_
#define MACHINA_LITE_TESTING_GENERATE_TESTSPEC_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "machina/lite/string_type.h"

namespace tflite {
namespace testing {

// Generate test spec by executing TensorFlow model on random inputs.
// The test spec can be consumed by ParseAndRunTests.
// See test spec format in parse_testdata.h
//
// Inputs:
//   stream: mutable iostream that contains the contents of test spec.
//   machina_model_path: path to TensorFlow model.
//   tflite_model_path: path to tflite_model_path that the test spec runs
//   num_invocations: how many pairs of inputs and outputs will be generated.
//   against. input_layer: names of input tensors. Example: input1
//   input_layer_type: datatypes of input tensors. Example: float
//   input_layer_shape: shapes of input tensors, separated by comma. example:
//   1,3,4 output_layer: names of output tensors. Example: output
bool GenerateTestSpecFromTensorflowModel(
    std::iostream& stream, const string& machina_model_path,
    const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer);

// Generate test spec by executing TFLite model on random inputs.
bool GenerateTestSpecFromTFLiteModel(
    std::iostream& stream, const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer);

// Generates random values that are filled into the tensor.
template <typename T, typename RandomFunction>
std::vector<T> GenerateRandomTensor(const std::vector<int>& shape,
                                    RandomFunction random_func) {
  int64_t num_elements = 1;
  for (const int dim : shape) {
    num_elements *= dim;
  }

  std::vector<T> result(num_elements);
  std::generate_n(result.data(), num_elements, random_func);
  return result;
}

}  // namespace testing
}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_GENERATE_TESTSPEC_H_
