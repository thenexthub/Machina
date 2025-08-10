/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/core/util/bad_indices_policy.h"

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/core/platform/test.h"

namespace machina {

namespace {  // Anonymous namespace to avoid name conflicts

constexpr absl::string_view kDefault = "DEFAULT";
constexpr absl::string_view kErrorStr = "ERROR";
constexpr absl::string_view kIgnoreStr = "IGNORE";

// Unit test class using Google Test framework
class BadIndicesPolicyFromStringTest : public ::testing::Test {
 protected:
  // Reusable function to test valid inputs
  void TestValidInput(absl::string_view input, BadIndicesPolicy expected) {
    absl::StatusOr<BadIndicesPolicy> result = BadIndicesPolicyFromString(input);
    ASSERT_TRUE(result.ok());             // Check for success
    EXPECT_EQ(result.value(), expected);  // Verify the policy value
  }
};

// Test cases covering valid inputs
TEST_F(BadIndicesPolicyFromStringTest, EmptyString) {
  TestValidInput("", BadIndicesPolicy::kDefault);
}

TEST_F(BadIndicesPolicyFromStringTest, DefaultKeyword) {
  TestValidInput(kDefault, BadIndicesPolicy::kDefault);
}

TEST_F(BadIndicesPolicyFromStringTest, ErrorKeyword) {
  TestValidInput(kErrorStr, BadIndicesPolicy::kError);
}

TEST_F(BadIndicesPolicyFromStringTest, IgnoreKeyword) {
  TestValidInput(kIgnoreStr, BadIndicesPolicy::kIgnore);
}

// Test case for invalid input
TEST_F(BadIndicesPolicyFromStringTest, InvalidInput) {
  absl::StatusOr<BadIndicesPolicy> result =
      BadIndicesPolicyFromString("unknown");
  ASSERT_FALSE(result.ok());  // Check for failure
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("Unknown bad indices handling attribute"));
}

}  // namespace

}  // namespace machina
