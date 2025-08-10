/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
// Testing TF2 enablement.

#include "machina/core/platform/enable_tf2_utils.h"

#include "machina/core/platform/test.h"
#include "machina/core/util/env_var.h"

namespace machina {

TEST(TF2EnabledTest, enabled_behavior) {
  string tf2_env;
  TF_CHECK_OK(ReadStringFromEnvVar("TF2_BEHAVIOR", "0", &tf2_env));
  bool expected = (tf2_env != "0");
  EXPECT_EQ(machina::tf2_execution_enabled(), expected);
  machina::set_tf2_execution(true);
  EXPECT_TRUE(machina::tf2_execution_enabled());
  machina::set_tf2_execution(false);
  EXPECT_FALSE(machina::tf2_execution_enabled());
}

}  // namespace machina
