/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/core/kernels/random_index_shuffle.h"

#include <array>
#include <vector>

#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace random {
namespace {

class RandomIndexShuffleTest : public ::testing::TestWithParam<uint64_t> {
 public:
  uint64_t GetMaxValue() const { return GetParam(); }
};

// Check that we do a correct bijection.
TEST_P(RandomIndexShuffleTest, Bijection) {
  const std::array<uint32, 3>& key = {42, 73, 1991};
  const uint64_t max_value = GetMaxValue();
  std::vector<bool> seen(max_value + 1, false);
  for (uint64_t value = 0; value <= max_value; ++value) {
    const uint64 output_value =
        index_shuffle(value, key, max_value, /* rounds= */ 4);
    EXPECT_GE(output_value, 0);
    EXPECT_LE(output_value, max_value);
    EXPECT_FALSE(seen[output_value]);
    seen[output_value] = true;
  }
}

INSTANTIATE_TEST_SUITE_P(MaxValueTests, RandomIndexShuffleTest,
                         ::testing::Values(0, 285, 17, 23495, 499'000));

}  // namespace
}  // namespace random
}  // namespace machina

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
