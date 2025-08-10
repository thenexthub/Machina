/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "machina/lite/experimental/shlo/dispatch.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/status_matcher.h"

namespace {

void VoidFunction() {}

TEST(DispatchTest, ReturnAbslOkIfVoidCompiles) {
  auto f = []() -> absl::Status { RETURN_OK_STATUS_IF_VOID(VoidFunction()); };
  EXPECT_OK(f());
}

TEST(DispatchTest, AbslOkStatusCompiles) {
  auto f = []() -> absl::Status { RETURN_OK_STATUS_IF_VOID(absl::OkStatus()); };
  EXPECT_OK(f());
}

TEST(DispatchTest, AbslErrorCompiles) {
  auto f = []() -> absl::Status {
    RETURN_OK_STATUS_IF_VOID(absl::UnknownError("error message"));
  };
  EXPECT_EQ(f(), absl::UnknownError("error message"));
}

}  // namespace
