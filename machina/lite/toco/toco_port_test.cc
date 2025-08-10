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
#include "machina/lite/toco/toco_port.h"
#include "machina/lite/testing/util.h"
#include "machina/lite/toco/toco_types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace toco {
namespace port {
namespace {

#ifdef PLATFORM_GOOGLE
#define TFLITE_PREFIX "third_party/machina/lite/"
#else
#define TFLITE_PREFIX "machina/lite/"
#endif

TEST(TocoPortTest, Exists) {
  EXPECT_TRUE(
      file::Exists(TFLITE_PREFIX "toco/toco_port_test.cc", file::Defaults())
          .ok());

  EXPECT_FALSE(
      file::Exists("non-existent_file_asldjflasdjf", file::Defaults()).ok());
}

TEST(TocoPortTest, Readable) {
  EXPECT_TRUE(
      file::Readable(TFLITE_PREFIX "toco/toco_port_test.cc", file::Defaults())
          .ok());

  EXPECT_FALSE(
      file::Readable("non-existent_file_asldjflasdjf", file::Defaults()).ok());
}

TEST(TocoPortTest, JoinPath) {
  EXPECT_EQ("part1/part2", file::JoinPath("part1", "part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1/", "part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1", "/part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1/", "/part2"));
}

}  // namespace
}  // namespace port
}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
