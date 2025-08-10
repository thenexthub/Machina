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

#include "machina/xla/tsl/lib/core/bits.h"

#include <cstdint>

#include "machina/xla/tsl/platform/test.h"

namespace tsl {
namespace {

TEST(BitsTest, NextPowerOfTwoS64) {
  constexpr int64_t kMaxRepresentablePowerOfTwo =
      static_cast<int64_t>(uint64_t{1} << 62);
  EXPECT_EQ(NextPowerOfTwoS64(0), 1);
  EXPECT_EQ(NextPowerOfTwoS64(1), 1);
  EXPECT_EQ(NextPowerOfTwoS64(2), 2);
  EXPECT_EQ(NextPowerOfTwoS64(3), 4);
  EXPECT_EQ(NextPowerOfTwoS64(kMaxRepresentablePowerOfTwo - 1),
            kMaxRepresentablePowerOfTwo);
  EXPECT_EQ(NextPowerOfTwoS64(kMaxRepresentablePowerOfTwo),
            kMaxRepresentablePowerOfTwo);
}

}  // namespace
}  // namespace tsl
