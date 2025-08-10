/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/lite/experimental/acceleration/compatibility/canonicalize_value.h"

#include <string>

#include <gtest/gtest.h>
#include "machina/lite/experimental/acceleration/compatibility/variables.h"

namespace tflite::acceleration {
namespace {

TEST(CanonicalizeValue, CharactersAreLowercased) {
  EXPECT_EQ(CanonicalizeValue("hElLo"), "hello");
}

TEST(CanonicalizeValue, HyphensAreReplaced) {
  EXPECT_EQ(CanonicalizeValue("-"), "_");
}

TEST(CanonicalizeValue, SpacesAreReplaced) {
  EXPECT_EQ(CanonicalizeValue(" "), "_");
}

TEST(CanonicalizeValue, OtherSpecialCharactersAreUnaffected) {
  for (unsigned char c = 0; c < 65; ++c) {
    if (c == ' ' || c == '-') continue;
    std::string s = {1, static_cast<char>(c)};
    EXPECT_EQ(CanonicalizeValue(s), s);
  }
}

TEST(CanonicalizeValue, SamsungXclipseGpuNormalized) {
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel, "ANGLE (Samsung Xclipse 920) on Vulkan 1.1.179"),
            "angle_(samsung_xclipse_920)_on_vulkan");
}
}  // namespace
}  // namespace tflite::acceleration
