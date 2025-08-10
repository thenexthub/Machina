/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "tsl/platform/stringpiece.h"

#include <unordered_map>

#include "machina/xla/tsl/platform/test.h"

namespace tsl {

TEST(StringPiece, Ctor) {
  {
    // const char* without size.
    const char* hello = "hello";
    absl::string_view s20(hello);
    EXPECT_TRUE(s20.data() == hello);
    EXPECT_EQ(5, s20.size());

    // const char* with size.
    absl::string_view s21(hello, 4);
    EXPECT_TRUE(s21.data() == hello);
    EXPECT_EQ(4, s21.size());

    // Not recommended, but valid C++
    absl::string_view s22(hello, 6);
    EXPECT_TRUE(s22.data() == hello);
    EXPECT_EQ(6, s22.size());
  }

  {
    string hola = "hola";
    absl::string_view s30(hola);
    EXPECT_TRUE(s30.data() == hola.data());
    EXPECT_EQ(4, s30.size());

    // std::string with embedded '\0'.
    hola.push_back('\0');
    hola.append("h2");
    hola.push_back('\0');
    absl::string_view s31(hola);
    EXPECT_TRUE(s31.data() == hola.data());
    EXPECT_EQ(8, s31.size());
  }
}

TEST(StringPiece, ConversionToString) {
  EXPECT_EQ("", string(absl::string_view("")));
  EXPECT_EQ("foo", string(absl::string_view("foo")));
}

}  // namespace tsl
