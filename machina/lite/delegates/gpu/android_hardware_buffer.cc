/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/lite/delegates/gpu/android_hardware_buffer.h"

#include <dlfcn.h>

namespace tflite::gpu {

OptionalAndroidHardwareBuffer::OptionalAndroidHardwareBuffer() {
#ifdef __ANDROID__
  dlopen_handle_ = dlopen("libnativewindow.so", RTLD_NOW);
  if (dlopen_handle_ == nullptr) {
    supported_ = false;
    return;
  }
  allocate_ = reinterpret_cast<decltype(allocate_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_allocate"));
  acquire_ = reinterpret_cast<decltype(acquire_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_acquire"));
  release_ = reinterpret_cast<decltype(release_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_release"));
  describe_ = reinterpret_cast<decltype(describe_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_describe"));
  is_supported_ = reinterpret_cast<decltype(is_supported_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_isSupported"));
  supported_ =
      (allocate_ != nullptr && acquire_ != nullptr && release_ != nullptr &&
       describe_ != nullptr && is_supported_ != nullptr);
#else
  dlopen_handle_ = nullptr;
  allocate_ = nullptr;
  acquire_ = nullptr;
  release_ = nullptr;
  describe_ = nullptr;
  is_supported_ = nullptr;
  supported_ = false;
#endif
}

}  // namespace tflite::gpu
