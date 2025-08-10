/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/lite/micro/micro_log.h"

#include <cstddef>
#include <cstring>

#include "machina/lite/micro/testing/micro_test.h"

namespace {

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
constexpr int kMaxBufferSize = 128;
const char* kFormat = "%2d%6.2f%#5x%5s";
const char* kExpect = "42 42.42 0x42 \"42\"";
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)

TF_LITE_MICRO_TEST(MicroPrintfTest) {
  MicroPrintf("Integer 42: %d", 42);
  MicroPrintf("Float 42.42: %2.2f", 42.42);
  MicroPrintf("String \"Hello World!\": %s", "\"Hello World!\"");
  MicroPrintf("Badly-formed format string %");
  MicroPrintf("Another %# badly-formed %% format string");
}

TF_LITE_MICRO_TEST(MicroSnprintf) {
  char buffer[kMaxBufferSize];
  buffer[0] = '\0';
  size_t result =
      MicroSnprintf(buffer, kMaxBufferSize, kFormat, 42, 42.42, 0x42, "\"42\"");
  TF_LITE_MICRO_EXPECT_EQ(result, strlen(buffer));
  TF_LITE_MICRO_EXPECT_STRING_EQ(kExpect, buffer);
}

#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)

TF_LITE_MICRO_TESTS_END
