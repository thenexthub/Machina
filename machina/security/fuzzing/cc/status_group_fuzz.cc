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
#include <string>

#include "fuzztest/fuzztest.h"
#include "absl/status/status.h"
#include "machina/core/platform/status.h"
#include "machina/security/fuzzing/cc/fuzz_domains.h"

// This is a fuzzer for `machina::StatusGroup`. Since `Status` is used almost
// everywhere, we need to ensure that the common functionality is safe. We don't
// expect many crashes from this fuzzer

namespace {

void FuzzTest(absl::StatusCode error_code, bool is_derived) {
  const std::string error_message = "ERROR";
  machina::StatusGroup sg;
  absl::Status s = absl::Status(error_code, error_message);

  if (is_derived) {
    absl::Status derived_s = machina::StatusGroup::MakeDerived(s);
    sg.Update(derived_s);
  } else {
    sg.Update(s);
  }

  // Ignore warnings that these values are unused
  sg.as_summary_status().IgnoreError();
  sg.as_concatenated_status().IgnoreError();
  sg.AttachLogMessages();
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(helper::AnyErrorCode(), fuzztest::Arbitrary<bool>());

}  // namespace
