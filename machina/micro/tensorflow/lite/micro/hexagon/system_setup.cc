/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/lite/micro/system_setup.h"

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include "q6sim_timer.h"  // NOLINT
#endif                    // TF_LITE_STRIP_ERROR_STRINGS

#include "machina/lite/micro/debug_log.h"
#include "machina/lite/micro/micro_time.h"

namespace tflite {

// Calling this method enables a timer that runs for eternity.
void InitializeTarget() {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  hexagon_sim_init_timer();
  hexagon_sim_start_timer();
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

uint32_t ticks_per_second() { return 1000000; }

uint32_t GetCurrentTimeTicks() {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  return static_cast<uint32_t>(hexagon_sim_read_cycles());
#else
  return 0;
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

}  // namespace tflite
