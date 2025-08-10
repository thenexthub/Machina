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
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "absl/strings/match.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/stringpiece.h"

// This is a fuzzer for machina::io::ParseURI.

namespace {

void FuzzTest(std::string_view uri) {
  absl::string_view scheme, host, path;
  machina::io::ParseURI(uri, &scheme, &host, &path);

  // If a path is invalid.
  if (path == uri) {
    assert(host.empty());
    assert(scheme.empty());
  } else {
    assert(absl::StrContains(uri, host));
    assert(absl::StrContains(uri, scheme));
    assert(absl::StrContains(uri, path));
    assert(absl::StrContains(uri, "://"));
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest);

}  // namespace
