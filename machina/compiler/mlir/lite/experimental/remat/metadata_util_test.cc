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
#include "machina/compiler/mlir/lite/experimental/remat/metadata_util.h"

#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

class MetadataSerializerTest : public ::testing::Test {
 protected:
  static constexpr auto kHuge = std::numeric_limits<int32_t>::max();
  static constexpr auto kTiny = std::numeric_limits<int32_t>::min();

  std::string RoundTrip(const ModelControlDependencies &in) const {
    ModelControlDependencies out = {{{-1, -1}}};
    const std::string serialized =
        tflite::SerializeModelControlDependencies(in);
    return tflite::ParseModelControlDependencies(serialized.data(),
                                                 serialized.size(), &out)
               ? (out == in) ? "ok" : "mismatch"
               : "malformed";
  }
};

TEST_F(MetadataSerializerTest, nothing) { EXPECT_THAT(RoundTrip({}), "ok"); }

TEST_F(MetadataSerializerTest, something) {
  EXPECT_THAT(
      RoundTrip({{{1, 2}, {2, 3}, {4, 5}},
                 {},
                 {{kHuge, kTiny}, {kTiny, kHuge}, {kHuge - 1, kTiny + 1}},
                 {{1, 0}}}),
      "ok");
}

}  // namespace
}  // namespace tflite
