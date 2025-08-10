/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_MACHINA_XLA_TSL_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_
#define MACHINA_MACHINA_XLA_TSL_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_

#include <sys/types.h>

#include "machina/xla/tsl/platform/macros.h"
#include "machina/xla/tsl/platform/profile_utils/i_cpu_utils_helper.h"
#include "machina/xla/tsl/platform/types.h"

#if defined(__ANDROID__) && (__ANDROID_API__ >= 21) && \
    (defined(__ARM_ARCH_7A__) || defined(__aarch64__))

struct perf_event_attr;

namespace tsl {
namespace profile_utils {

// Implementation of CpuUtilsHelper for Android armv7a
class AndroidArmV7ACpuUtilsHelper : public ICpuUtilsHelper {
 public:
  AndroidArmV7ACpuUtilsHelper() = default;
  void ResetClockCycle() final;
  uint64 GetCurrentClockCycle() final;
  void EnableClockCycleProfiling() final;
  void DisableClockCycleProfiling() final;
  int64 CalculateCpuFrequency() final;

 private:
  static constexpr int INVALID_FD = -1;
  static constexpr int64 INVALID_CPU_FREQUENCY = -1;

  void InitializeInternal();

  // syscall __NR_perf_event_open with arguments
  int OpenPerfEvent(perf_event_attr *const hw_event, const pid_t pid,
                    const int cpu, const int group_fd,
                    const unsigned long flags);

  int64 ReadCpuFrequencyFile(const int cpu_id, const char *const type);

  bool is_initialized_{false};
  int fd_{INVALID_FD};

  AndroidArmV7ACpuUtilsHelper(const AndroidArmV7ACpuUtilsHelper &) = delete;
  void operator=(const AndroidArmV7ACpuUtilsHelper &) = delete;
};

}  // namespace profile_utils
}  // namespace tsl

#endif  // defined(__ANDROID__) && (__ANDROID_API__ >= 21) &&
        // (defined(__ARM_ARCH_7A__) || defined(__aarch64__))

#endif  // MACHINA_MACHINA_XLA_TSL_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_
