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

#include "machina/compiler/jit/device_compiler_client.h"

#include <gtest/gtest.h>

namespace machina {
namespace {

TEST(GetExecutableOptionTest, Basic) {
  XlaCompiler::Options options;
  options.device_ordinal = 0;
  options.alias_passthrough_params = true;
  options.detailed_logging = true;
  XlaCompiler::CompilationResult result;
  xla::Shape xla_output_shape;
  result.xla_output_shape = xla_output_shape;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_EQ(build_option.device_ordinal(), 0);
  EXPECT_EQ(build_option.result_layout()->ToString(),
            xla_output_shape.ToString());
  EXPECT_EQ(build_option.alias_passthrough_params(), true);
  EXPECT_EQ(build_option.debug_options().xla_detailed_logging(), true);
  EXPECT_EQ(build_option.debug_options().xla_enable_dumping(), true);
}

TEST(GetExecutableOptionTest, DefaultDeviceOrdinal) {
  XlaCompiler::Options options;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/0);

  EXPECT_EQ(build_option.device_ordinal(), 0);
}

TEST(GetExecutableOptionTest, DeviceOrdinalNotSet) {
  XlaCompiler::Options options;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_EQ(build_option.device_ordinal(), -1);
}

TEST(GetExecutableOptionTest, DumpingWithoutDetailedLogging) {
  XlaCompiler::Options options;
  options.detailed_logging = false;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_FALSE(build_option.debug_options().xla_detailed_logging());
  EXPECT_TRUE(build_option.debug_options().xla_enable_dumping());
}

}  // namespace
}  // namespace machina
