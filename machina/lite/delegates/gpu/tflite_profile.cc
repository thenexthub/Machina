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
#include "machina/lite/delegates/gpu/tflite_profile.h"

#include "absl/time/time.h"
#include "machina/lite/core/api/profiler.h"
#include "machina/lite/delegates/gpu/common/task/profiling_info.h"

namespace tflite {
namespace gpu {

static void* s_profiler = nullptr;

bool IsTfLiteProfilerActive() { return s_profiler != nullptr; }

void SetTfLiteProfiler(void* profiler) { s_profiler = profiler; }

void* GetTfLiteProfiler() { return s_profiler; }

void AddTfLiteProfilerEvents(tflite::gpu::ProfilingInfo* profiling_info) {
  tflite::Profiler* profile =
      reinterpret_cast<tflite::Profiler*>(GetTfLiteProfiler());
  if (profile == nullptr) return;

  int node_index = 0;
  for (const auto& dispatch : profiling_info->dispatches) {
    profile->AddEvent(
        dispatch.label.c_str(),
        Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT,
        absl::ToDoubleMicroseconds(dispatch.duration), node_index++);
  }
}

}  // namespace gpu
}  // namespace tflite
