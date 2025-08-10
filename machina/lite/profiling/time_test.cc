/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/lite/profiling/time.h"
#include <gtest/gtest.h>

namespace tflite {
namespace profiling {
namespace time {

TEST(TimeTest, NowMicros) {
  auto now0 = NowMicros();
  EXPECT_GT(now0, 0);
  auto now1 = NowMicros();
  EXPECT_GE(now1, now0);
}

TEST(TimeTest, SleepForMicros) {
  // A zero sleep shouldn't cause issues.
  SleepForMicros(0);

  // Sleeping should be reflected in the current time.
  auto now0 = NowMicros();
  SleepForMicros(50);
  auto now1 = NowMicros();
  EXPECT_GE(now1, now0 + 50);

  // Sleeping more than a second should function properly.
  now0 = NowMicros();
  SleepForMicros(1e6 + 50);
  now1 = NowMicros();
  EXPECT_GE(now1, now0 + 1e6 + 50);
}

}  // namespace time
}  // namespace profiling
}  // namespace tflite
