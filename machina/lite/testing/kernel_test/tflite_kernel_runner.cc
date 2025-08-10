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

#include <memory>

#include "machina/lite/testing/kernel_test/util.h"

int main(int argc, char** argv) {
  tflite::testing::kernel_test::TestOptions options =
      tflite::testing::kernel_test::ParseTfliteKernelTestFlags(&argc, argv);
  const bool run_reference_kernel = options.kernel_type == "REFERENCE";
  const tflite::testing::TfLiteDriver::DelegateType delegate_type =
      options.kernel_type == "NNAPI"
          ? tflite::testing::TfLiteDriver::DelegateType::kNnapi
          : tflite::testing::TfLiteDriver::DelegateType::kNone;

  auto runner = std::make_unique<tflite::testing::TfLiteDriver>(
      delegate_type, run_reference_kernel);
  if (tflite::testing::kernel_test::RunKernelTest(options, runner.get()) ==
      kTfLiteOk) {
    return 0;
  }

  return -1;
}
