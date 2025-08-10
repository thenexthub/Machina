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
#ifndef MACHINA_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_
#define MACHINA_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "machina/lite/core/c/common.h"
#include "machina/lite/string_type.h"

namespace tflite {
namespace testing {

// Reads the baseline and test files with output tensor values, and calculates
// the diff metrics.
class DiffAnalyzer {
 public:
  DiffAnalyzer() = default;
  // Reads base and test tensor values from files.
  // Each file have lines in <name>:<values> format, where name is the signature
  // output name and value as comma separated value string.
  TfLiteStatus ReadFiles(const string& base, const string& test);
  // Writes diff report in <name>:<L2 Error>,<Max Diff> format.
  TfLiteStatus WriteReport(const string& filename);

 private:
  // Mappings from signature output names to its values.
  std::unordered_map<string, std::vector<float>> base_tensors_;
  std::unordered_map<string, std::vector<float>> test_tensors_;
};

}  // namespace testing
}  // namespace tflite

#endif  // MACHINA_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_
