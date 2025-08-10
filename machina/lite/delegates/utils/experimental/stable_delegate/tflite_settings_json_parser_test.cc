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
#include "machina/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "machina/lite/acceleration/configuration/configuration_generated.h"

namespace {

using tflite::TFLiteSettings;
using tflite::delegates::utils::TfLiteSettingsJsonParser;

TEST(TfLiteSettingsJsonParserTest, SuccessWithValidXNNPackDelegateSettings) {
  TfLiteSettingsJsonParser parser;

  const TFLiteSettings* tflite_settings = parser.Parse(
      "machina/lite/delegates/utils/experimental/"
      "stable_delegate/test_xnnpack_settings.json");

  EXPECT_NE(parser.GetBufferPointer(), nullptr);
  EXPECT_NE(parser.GetBufferSize(), 0);
  ASSERT_NE(tflite_settings, nullptr);
  EXPECT_EQ(tflite_settings->delegate(), tflite::Delegate_XNNPACK);
  ASSERT_NE(tflite_settings->xnnpack_settings(), nullptr);
  EXPECT_EQ(tflite_settings->xnnpack_settings()->num_threads(), 5);
}

TEST(TfLiteSettingsJsonParserTest, GetBufferPointerReturnsValidBufferPointers) {
  TfLiteSettingsJsonParser parser;
  parser.Parse(
      "machina/lite/delegates/utils/experimental/"
      "stable_delegate/test_xnnpack_settings.json");
  const uint8_t* buffer_pointer = parser.GetBufferPointer();

  ASSERT_NE(buffer_pointer, nullptr);
  ASSERT_NE(parser.GetBufferSize(), 0);
  const TFLiteSettings* tflite_settings =
      flatbuffers::GetRoot<TFLiteSettings>(buffer_pointer);
  ASSERT_NE(tflite_settings, nullptr);
  EXPECT_EQ(tflite_settings->delegate(), tflite::Delegate_XNNPACK);
  ASSERT_NE(tflite_settings->xnnpack_settings(), nullptr);
  EXPECT_EQ(tflite_settings->xnnpack_settings()->num_threads(), 5);
}

// This test passes the path to a JSON file that the content of the file cannot
// be parsed into the TFLiteSettings structure.
TEST(TfLiteSettingsJsonParserTest, FailedToParseInvalidSettings) {
  TfLiteSettingsJsonParser parser;

  EXPECT_EQ(
      parser.Parse("machina/lite/tools/delegates/experimental/"
                   "stable_delegate/test_invalid_settings.json"),
      nullptr);
  EXPECT_EQ(parser.GetBufferPointer(), nullptr);
  EXPECT_EQ(parser.GetBufferSize(), 0);
}

}  // namespace
