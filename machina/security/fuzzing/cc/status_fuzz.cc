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
#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "absl/status/status.h"
#include "machina/core/platform/status.h"
#include "machina/security/fuzzing/cc/fuzz_domains.h"

// This is a fuzzer for `machina::Status`. Since `Status` is used almost
// everywhere, we need to ensure that the common functionality is safe. We don't
// expect many crashes from this fuzzer since we only create a status and then
// look at the error message from it but this is a good test of the fuzzing
// infrastructure, with minimal dependencies (thus, it is a good test to weed
// out linker bloat and other linker issues).

namespace {

void FuzzTest(absl::StatusCode error_code, std::string_view error_message) {
  absl::Status s = absl::Status(error_code, error_message);
  const std::string actual_message = s.ToString();
  const std::size_t pos = actual_message.rfind(error_message);
  assert(pos != std::string::npos);  // Suffix is error message
  assert(pos > 0);                   // Prefix is error code

  // In some build configurations `assert` is a no-op. This causes `pos` to be
  // unused and then produces an error if also compiling with `-Werror`.
  (void)pos;
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(helper::AnyErrorCode(), fuzztest::Arbitrary<std::string>());

}  // namespace
