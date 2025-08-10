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
#include "machina/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"

#include <cstdlib>

#include <gtest/gtest.h>
#include "machina/lite/acceleration/configuration/c/stable_delegate.h"
#include "machina/lite/acceleration/configuration/configuration_generated.h"
#include "machina/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"

namespace {

using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
using tflite::delegates::utils::LoadDelegateFromSharedLibrary;
using tflite::delegates::utils::LoadSymbolFromSharedLibrary;

TEST(TfLiteDelegateLoaderUtilsTest, Simple) {
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary(
          "machina/lite/delegates/utils/experimental/"
          "sample_stable_delegate/"
          "libmachinalite_sample_stable_delegate.so"
          );

  ASSERT_NE(stable_delegate_handle, nullptr);
  EXPECT_STREQ(stable_delegate_handle->delegate_abi_version,
               TFL_STABLE_DELEGATE_ABI_VERSION);
  EXPECT_STREQ(stable_delegate_handle->delegate_name,
               tflite::example::kSampleStableDelegateName);
  EXPECT_STREQ(stable_delegate_handle->delegate_version,
               tflite::example::kSampleStableDelegateVersion);
  EXPECT_NE(stable_delegate_handle->delegate_plugin, nullptr);
  EXPECT_STREQ(
      getenv(tflite::delegates::utils::kTfLiteLibraryPathEnvironmentVariable),
      "machina/lite/delegates/utils/experimental/"
      "sample_stable_delegate");

  // Builds TFLiteSettings flatbuffer and passes into delegate plugin create
  // method.
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());
  auto delegate = stable_delegate_handle->delegate_plugin->create(settings);

  ASSERT_NE(delegate, nullptr);
  EXPECT_EQ(
      stable_delegate_handle->delegate_plugin->get_delegate_errno(delegate), 0);

  stable_delegate_handle->delegate_plugin->destroy(delegate);
}

TEST(TfLiteDelegateLoaderUtilsTest, WrongSymbolReturnsNullptr) {
  void* symbol_pointer = LoadSymbolFromSharedLibrary(
      "machina/lite/delegates/utils/experimental/"
      "sample_stable_delegate/libmachinalite_sample_stable_delegate.so",
      "NOT_REAL_SYMBOL");
  EXPECT_EQ(symbol_pointer, nullptr);
}

TEST(TfLiteDelegateLoaderUtilsTest, MissingLibReturnsNullptr) {
  const TfLiteStableDelegate* stable_delegate_handle =
      LoadDelegateFromSharedLibrary("not_real_delegate.so");
  EXPECT_EQ(stable_delegate_handle, nullptr);
}

}  // namespace
