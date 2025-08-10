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
#include <string>

#include "fuzztest/fuzztest.h"
#include "machina/core/platform/str_util.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"

// This is a fuzzer for machina::str_util::ConsumeLeadingDigits

namespace {

void FuzzTest(std::string data) {
  absl::string_view sp(data);
  machina::uint64 val;

  const bool leading_digits =
      machina::str_util::ConsumeLeadingDigits(&sp, &val);
  const char lead_char_consume_digits = *(sp.data());
  if (leading_digits) {
    if (lead_char_consume_digits >= '0') {
      assert(lead_char_consume_digits > '9');
    }
    assert(val >= 0);
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(
        fuzztest::Arbitrary<std::string>().WithMaxSize(25));
}  // namespace
