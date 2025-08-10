/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "machina/lite/testing/join.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

TEST(JoinTest, JoinInt) {
  std::vector<int> data = {1, 2, 3};
  EXPECT_EQ(Join(data.data(), data.size(), ","), "1,2,3");
}

TEST(JoinDefaultTest, JoinFloat) {
  float data[] = {1.0, -3, 2.3, 1e-5};
  EXPECT_EQ(JoinDefault(data, 4, " "), "1 -3 2.3 1e-05");
}

TEST(JoinTest, JoinFloat) {
  float data[] = {1.0, -3, 2.3, 1e-5};
  EXPECT_EQ(Join(data, 4, " "), "1 -3 2.29999995 9.99999975e-06");
}

TEST(JoinTest, JoinNullData) { EXPECT_THAT(Join<int>(nullptr, 3, ","), ""); }

TEST(JoinTest, JoinZeroData) {
  std::vector<int> data;
  EXPECT_THAT(Join(data.data(), 0, ","), "");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
