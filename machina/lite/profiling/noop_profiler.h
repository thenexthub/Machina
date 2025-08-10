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
#ifndef MACHINA_LITE_PROFILING_NOOP_PROFILER_H_
#define MACHINA_LITE_PROFILING_NOOP_PROFILER_H_

#include <vector>

#include "machina/lite/core/api/profiler.h"
#include "machina/lite/profiling/profile_buffer.h"

namespace tflite {
namespace profiling {

// A noop version of profiler when profiling is disabled.
class NoopProfiler : public tflite::Profiler {
 public:
  NoopProfiler() {}
  explicit NoopProfiler(int max_profiling_buffer_entries) {}

  uint32_t BeginEvent(const char*, EventType, int64_t, int64_t) override {
    return 0;
  }
  void EndEvent(uint32_t) override {}

  void StartProfiling() {}
  void StopProfiling() {}
  void Reset() {}
  std::vector<const ProfileEvent*> GetProfileEvents() { return {}; }
};

}  // namespace profiling
}  // namespace tflite

#endif  // MACHINA_LITE_PROFILING_NOOP_PROFILER_H_
