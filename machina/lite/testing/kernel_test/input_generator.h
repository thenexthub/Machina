/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_
#define MACHINA_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/core/interpreter.h"
#include "machina/lite/core/model.h"
#include "machina/lite/signature_runner.h"
#include "machina/lite/string_type.h"

namespace tflite {
namespace testing {

// Generate random input, or read input from a file for kernel diff test.
// Needs to load the tflite graph to get information like tensor shape and
// data type.
class InputGenerator {
 public:
  InputGenerator() = default;
  TfLiteStatus LoadModel(const string& model_dir);
  TfLiteStatus LoadModel(const string& model_dir, const string& signature);
  TfLiteStatus ReadInputsFromFile(const string& filename);
  TfLiteStatus GenerateInput(const string& distribution);
  std::vector<std::pair<string, string>> GetInputs() { return inputs_; }
  TfLiteStatus WriteInputsToFile(const string& filename);

 private:
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  // Not owned.
  SignatureRunner* signature_runner_ = nullptr;
  // Mapping from input names to csv string values.
  std::vector<std::pair<string, string>> inputs_;
};

}  // namespace testing
}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_
