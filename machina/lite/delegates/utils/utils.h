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
#ifndef MACHINA_LITE_DELEGATES_UTILS_UTILS_H_
#define MACHINA_LITE_DELEGATES_UTILS_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/delegates/utils/ret_macros.h"

namespace tflite::delegates::utils {

// Returns kTfLiteOk if the status is ok;
// Otherwise, log the error message and returns kTfLiteError.
TfLiteStatus ConvertToTfLiteStatus(absl::Status status);

inline bool IsPowerOfTwo(size_t x) { return x && ((x & (x - 1)) == 0); }

// Round up "size" to the nearest multiple of "multiple".
// "multiple" must be a power of 2.
inline uint32_t RoundUp(uint32_t size, uint32_t multiple) {
  TFLITE_ABORT_CHECK(IsPowerOfTwo(multiple), "");  // Crash OK
  return (size + (multiple - 1)) & ~(multiple - 1);
}

}  // namespace tflite::delegates::utils

#endif  // MACHINA_LITE_DELEGATES_UTILS_UTILS_H_
