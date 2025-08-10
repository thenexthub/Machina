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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_COMPILATION_TIMER_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_COMPILATION_TIMER_H_

#include <chrono>  // NOLINT(build/c++11)

#include "machina/core/platform/profile_utils/cpu_utils.h"

// Time the execution of kernels (in CPU cycles). Meant to be used as RAII.
struct CompilationTimer {
  uint64_t start_cycles =
      machina::profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64_t ElapsedCycles() {
    return machina::profile_utils::CpuUtils::GetCurrentClockCycle() -
           start_cycles;
  }

  int64_t ElapsedCyclesInMilliseconds() {
    std::chrono::duration<double> duration =
        machina::profile_utils::CpuUtils::ConvertClockCycleToTime(
            ElapsedCycles());

    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_COMPILATION_TIMER_H_
