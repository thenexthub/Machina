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

#include "machina/examples/speech_commands/accuracy_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {

TEST(AccuracyUtilsTest, ReadGroundTruthFile) {
  string file_name = machina::io::JoinPath(machina::testing::TmpDir(),
                                              "ground_truth.txt");
  string file_data = "a,10\nb,12\n";
  TF_ASSERT_OK(WriteStringToFile(Env::Default(), file_name, file_data));

  std::vector<std::pair<string, int64_t>> ground_truth;
  TF_ASSERT_OK(ReadGroundTruthFile(file_name, &ground_truth));
  ASSERT_EQ(2, ground_truth.size());
  EXPECT_EQ("a", ground_truth[0].first);
  EXPECT_EQ(10, ground_truth[0].second);
  EXPECT_EQ("b", ground_truth[1].first);
  EXPECT_EQ(12, ground_truth[1].second);
}

TEST(AccuracyUtilsTest, CalculateAccuracyStats) {
  StreamingAccuracyStats stats;
  CalculateAccuracyStats({{"a", 1000}, {"b", 9000}},
                         {{"a", 1200}, {"b", 5000}, {"a", 8700}}, 10000, 500,
                         &stats);
  EXPECT_EQ(2, stats.how_many_ground_truth_words);
  EXPECT_EQ(2, stats.how_many_ground_truth_matched);
  EXPECT_EQ(1, stats.how_many_false_positives);
  EXPECT_EQ(1, stats.how_many_correct_words);
  EXPECT_EQ(1, stats.how_many_wrong_words);
}

TEST(AccuracyUtilsTest, PrintAccuracyStats) {
  StreamingAccuracyStats stats;
  PrintAccuracyStats(stats);
}

}  // namespace machina
