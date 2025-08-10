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
#ifndef MACHINA_LITE_DELEGATES_GPU_TFLITE_PROFILE_H_
#define MACHINA_LITE_DELEGATES_GPU_TFLITE_PROFILE_H_

#include "machina/lite/delegates/gpu/common/task/profiling_info.h"

namespace tflite {
namespace gpu {

// Returns if TFLite Profiler is active.
bool IsTfLiteProfilerActive();

// Save the given TFLite Profiler object (from TfLiteContext) for op profiling.
void SetTfLiteProfiler(void* profiler);

// Returns saved TFLite Profiler object.
void* GetTfLiteProfiler();

// Generate TFLite Profiler events with the given ProfilingInfo object.
void AddTfLiteProfilerEvents(tflite::gpu::ProfilingInfo* profiling_info);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_TFLITE_PROFILE_H_
