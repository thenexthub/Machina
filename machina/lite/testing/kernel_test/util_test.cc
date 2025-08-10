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
#include "machina/lite/testing/kernel_test/util.h"

#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "machina/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace kernel_test {
namespace {

TEST(UtilTest, SimpleE2ETest) {
  TestOptions options;
  options.tflite_model = "machina/lite/testdata/add.bin";
  options.read_input_from_file =
      "machina/lite/testing/kernel_test/testdata/test_input.csv";
  options.dump_output_to_file = ::testing::TempDir() + "/test_out.csv";
  options.kernel_type = "REFERENCE";
  std::unique_ptr<TestRunner> runner(new TfLiteDriver(
      TfLiteDriver::DelegateType::kNone, /*reference_kernel=*/true));
  RunKernelTest(options, runner.get());
  std::string expected = "x:3";
  for (int i = 0; i < 1 * 8 * 8 * 3 - 1; i++) {
    expected.append(",3");
  }
  std::string content;
  std::ifstream file(options.dump_output_to_file);
  std::getline(file, content);
  EXPECT_EQ(content, expected);
}

}  // namespace
}  // namespace kernel_test
}  // namespace testing
}  // namespace tflite
