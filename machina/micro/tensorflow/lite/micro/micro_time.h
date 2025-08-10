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
#ifndef MACHINA_LITE_MICRO_MICRO_TIME_H_
#define MACHINA_LITE_MICRO_MICRO_TIME_H_

#include <cstdint>

namespace tflite {

// These functions should be implemented by each target platform, and provide an
// accurate tick count along with how many ticks there are per second.
uint32_t ticks_per_second();

// Return time in ticks.  The meaning of a tick varies per platform.
uint32_t GetCurrentTimeTicks();

inline uint32_t TicksToMs(int32_t ticks) {
  uint32_t _ticks_per_second = ticks_per_second();
  _ticks_per_second =
      _ticks_per_second > 0 ? _ticks_per_second : 1;  // zero divide prevention
  return static_cast<uint32_t>(1000.0f * static_cast<float>(ticks) /
                               static_cast<float>(_ticks_per_second));
}

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_MICRO_TIME_H_
