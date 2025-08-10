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

#include "machina/core/config/flags.h"

#include "machina/core/config/flag_defs.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

TEST(TFFlags, ReadFlagValue) {
  EXPECT_TRUE(flags::Global().test_only_experiment_1.value());
  EXPECT_FALSE(flags::Global().test_only_experiment_2.value());
}

TEST(TFFlags, ResetFlagValue) {
  EXPECT_TRUE(flags::Global().test_only_experiment_1.value());
  flags::Global().test_only_experiment_1.reset(false);
  EXPECT_FALSE(flags::Global().test_only_experiment_1.value());
}

}  // namespace
}  // namespace machina
