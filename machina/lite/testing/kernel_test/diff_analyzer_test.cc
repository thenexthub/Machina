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
#include "machina/lite/testing/kernel_test/diff_analyzer.h"

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include "machina/core/lib/io/path.h"

namespace tflite {
namespace testing {

namespace {

TEST(DiffAnalyzerTest, ZeroDiff) {
  DiffAnalyzer diff_analyzer;
  string filename =
      "machina/lite/testing/kernel_test/testdata/test_input.csv";
  ASSERT_EQ(diff_analyzer.ReadFiles(filename, filename), kTfLiteOk);

  string output_file =
      machina::io::JoinPath(::testing::TempDir(), "diff_report.csv");
  ASSERT_EQ(diff_analyzer.WriteReport(output_file), kTfLiteOk);

  std::string content;
  std::ifstream file(output_file);
  std::getline(file, content);
  std::getline(file, content);
  ASSERT_EQ(content, "a:0,0");
}

}  // namespace

}  // namespace testing
}  // namespace tflite
