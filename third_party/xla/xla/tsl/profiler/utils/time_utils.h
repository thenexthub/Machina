/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_XLATSL_PROFILER_UTILS_TIME_UTILS_H_
#define MACHINA_XLATSL_PROFILER_UTILS_TIME_UTILS_H_

#include <cstdint>

#include "machina/xla/tsl/profiler/utils/math_utils.h"

namespace tsl {
namespace profiler {

// Returns the current CPU wallclock time in nanoseconds.
int64_t GetCurrentTimeNanos();

// Sleeps for the specified duration.
void SleepForNanos(int64_t ns);
inline void SleepForMicros(int64_t us) { SleepForNanos(MicroToNano(us)); }
inline void SleepForMillis(int64_t ms) { SleepForNanos(MilliToNano(ms)); }
inline void SleepForSeconds(int64_t s) { SleepForNanos(UniToNano(s)); }

// Spins to simulate doing some work instead of sleeping, because sleep
// precision is poor. For testing only.
void SpinForNanos(int64_t ns);
inline void SpinForMicros(int64_t us) { SpinForNanos(us * 1000); }

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_TIME_UTILS_H_
