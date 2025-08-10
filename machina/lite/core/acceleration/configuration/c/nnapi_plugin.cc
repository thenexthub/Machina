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

// This file implements the Delegate Plugin for the NNAPI Delegate.

#include "machina/lite/core/acceleration/configuration/c/nnapi_plugin.h"

#include "machina/lite/acceleration/configuration/configuration_generated.h"
#include "machina/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "machina/lite/core/acceleration/configuration/nnapi_plugin.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/nnapi/nnapi_delegate.h"

extern "C" {

static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  tflite::delegates::NnapiPlugin nnapi_plugin(*tflite_settings);
  auto support_library_handle = nnapi_plugin.GetSupportLibraryHandle();
  if (support_library_handle) {
    auto nnapi_support_library_driver =
        reinterpret_cast<const NnApiSLDriverImplFL5*>(support_library_handle);
    return new tflite::StatefulNnApiDelegate(nnapi_support_library_driver,
                                             nnapi_plugin.Options());
  }
  return new tflite::StatefulNnApiDelegate(nnapi_plugin.Options());
}

static void DestroyDelegate(TfLiteDelegate* delegate) {
  delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
}

static int DelegateErrno(TfLiteDelegate* from_delegate) {
  auto nnapi_delegate =
      static_cast<tflite::StatefulNnApiDelegate*>(from_delegate);
  return nnapi_delegate->GetNnApiErrno();
}

static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};

const TfLiteDelegatePlugin* TfLiteNnapiDelegatePluginCApi() {
  return &kPluginCApi;
}

}  // extern "C"
