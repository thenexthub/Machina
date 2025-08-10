
/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/lite/schema/builtin_ops_header/generator.h"

#include <gtest/gtest.h>

namespace {

using tflite::builtin_ops_header::ConstantizeVariableName;
using tflite::builtin_ops_header::IsValidInputEnumName;

TEST(TestIsValidInputEnumName, TestWithValidInputNames) {
  EXPECT_TRUE(IsValidInputEnumName("ADD"));
  EXPECT_TRUE(IsValidInputEnumName("CONV_2D"));
  EXPECT_TRUE(IsValidInputEnumName("L2_POOL_2D"));
}

TEST(TestIsValidInputEnumName, TestWithLeadingUnderscore) {
  EXPECT_FALSE(IsValidInputEnumName("_ADD"));
  EXPECT_FALSE(IsValidInputEnumName("_CONV_2D"));
}

TEST(TestIsValidInputEnumName, TestWithLowerCase) {
  EXPECT_FALSE(IsValidInputEnumName("_AdD"));
  EXPECT_FALSE(IsValidInputEnumName("_COnV_2D"));
}

TEST(TestIsValidInputEnumName, TestWithOtherCharacters) {
  EXPECT_FALSE(IsValidInputEnumName("_AdD!2D"));
  EXPECT_FALSE(IsValidInputEnumName("_COnV?2D"));
}

TEST(TestIsValidInputEnumName, TestWithDoubleUnderscores) {
  EXPECT_FALSE(IsValidInputEnumName("ADD__2D"));
  EXPECT_FALSE(IsValidInputEnumName("CONV__2D"));
}

TEST(TestConstantizeVariableName, TestWithValidInputNames) {
  EXPECT_EQ(ConstantizeVariableName("ADD"), "kTfLiteBuiltinAdd");
  EXPECT_EQ(ConstantizeVariableName("CONV_2D"), "kTfLiteBuiltinConv2d");
  EXPECT_EQ(ConstantizeVariableName("L2_POOL_2D"), "kTfLiteBuiltinL2Pool2d");
}

}  // anonymous namespace
