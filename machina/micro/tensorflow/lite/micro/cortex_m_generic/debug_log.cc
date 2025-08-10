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

// Implementation for the DebugLog() function that prints to the debug logger on
// an generic Cortex-M device.

#include "machina/lite/micro/debug_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "machina/lite/micro/cortex_m_generic/debug_log_callback.h"

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include <stdio.h>
#endif

static DebugLogCallback debug_log_callback = nullptr;

namespace {

void InvokeDebugLogCallback(const char* s) {
  if (debug_log_callback != nullptr) {
    debug_log_callback(s);
  }
}

}  // namespace

void RegisterDebugLogCallback(void (*cb)(const char* s)) {
  debug_log_callback = cb;
}

void DebugLog(const char* format, va_list args) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];

  vsnprintf(log_buffer, kMaxLogLen, format, args);
  InvokeDebugLogCallback(log_buffer);
#endif
}

#ifndef TF_LITE_STRIP_ERROR_STRINGS
// Only called from MicroVsnprintf (micro_log.h)
int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
