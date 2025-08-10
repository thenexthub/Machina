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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "machina/lite/kernels/test_delegate_providers.h"
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/testing/util.h"
#include "machina/lite/tools/command_line_flags.h"

namespace {

bool InitKernelTest(int* argc, char** argv) {
  tflite::KernelTestDelegateProviders* const delegate_providers =
      tflite::KernelTestDelegateProviders::Get();
  if (!delegate_providers->InitFromCmdlineArgs(
          argc, const_cast<const char**>(argv))) {
    return false;
  }

  if (delegate_providers->ConstParams().Get<bool>("use_nnapi")) {
    // In Android Q, the NNAPI delegate avoids delegation if the only device
    // is the reference CPU. However, for testing purposes, we still want
    // delegation coverage, so force use of this reference path.
    auto* params = delegate_providers->MutableParams();
    if (!params->HasValueSet<std::string>("nnapi_accelerator_name")) {
      params->Set<std::string>("nnapi_accelerator_name", "nnapi-reference");
      params->Set("disable_nnapi_cpu", false);
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  if (InitKernelTest(&argc, argv)) {
    ::testing::InitGoogleTest(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return RUN_ALL_TESTS();
  } else {
    return EXIT_FAILURE;
  }
}
