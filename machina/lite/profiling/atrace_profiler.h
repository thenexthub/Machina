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
#ifndef MACHINA_LITE_PROFILING_ATRACE_PROFILER_H_
#define MACHINA_LITE_PROFILING_ATRACE_PROFILER_H_

#include <memory>

#include "machina/lite/core/api/profiler.h"

namespace tflite {
namespace profiling {

// Creates a profiler which reports the traced events to the Android ATrace.
// Nullptr will be returned if the Android system property 'debug.tflite.trace'
// is not set or the property value is not 1.
std::unique_ptr<tflite::Profiler> MaybeCreateATraceProfiler();

}  // namespace profiling
}  // namespace tflite

#endif  // MACHINA_LITE_PROFILING_ATRACE_PROFILER_H_
