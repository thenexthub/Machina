/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "machina/xla/tsl/platform/profile_utils/clock_cycle_profiler.h"

#include <chrono>

namespace tsl {

void ClockCycleProfiler::DumpStatistics(const string& tag) {
  CHECK(!IsStarted());
  const double average_clock_cycle = GetAverageClockCycle();
  const double count = GetCount();
  const std::chrono::duration<double> average_time =
      profile_utils::CpuUtils::ConvertClockCycleToTime(
          static_cast<int64_t>(average_clock_cycle + 0.5));
  LOG(INFO) << tag << ": average = "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   average_time)
                   .count()
            << " us (" << average_clock_cycle << " cycles)"
            << ", count = " << count;
}

}  // namespace tsl
