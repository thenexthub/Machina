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

#include "machina/core/tfrt/run_handler_thread_pool/run_handler_util.h"

#include <vector>

#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"

namespace tfrt {
namespace tf {
namespace {

TEST(RunHandlerUtilTest, TestParamFromEnvWithDefault) {
  std::vector<double> result = ParamFromEnvWithDefault(
      "RUN_HANDLER_TEST_ENV", std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 0);
  EXPECT_EQ(result[2], 0);

  std::vector<int> result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                                     std::vector<int>{0, 0, 0});
  EXPECT_EQ(result2.size(), 3);
  EXPECT_EQ(result2[0], 0);
  EXPECT_EQ(result2[1], 0);
  EXPECT_EQ(result2[2], 0);

  bool result3 =
      ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, false);

  // Set environment variable.
  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV", "1,2,3", true), 0);
  result = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                   std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
  result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                    std::vector<int>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result2[0], 1);
  EXPECT_EQ(result2[1], 2);
  EXPECT_EQ(result2[2], 3);

  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV_BOOL", "true", true), 0);
  result3 = ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, true);
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
