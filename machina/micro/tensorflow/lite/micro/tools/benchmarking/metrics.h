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

#ifndef TFLM_BENCHMARK_INTERNAL_METRICS_H_
#define TFLM_BENCHMARK_INTERNAL_METRICS_H_

#include "machina/lite/micro/micro_profiler.h"
#include "machina/lite/micro/recording_micro_allocator.h"

namespace tflite {

// Defines how formatted data is printed to stdout.
enum class PrettyPrintType {
  // Prints as a CSV file.
  kCsv,
  // Prints as a formatted table.
  kTable,
};

// Logs the allocation events. Prints out two tables, one for the arena
// allocations, and one for each type of TFLM allocation type.
// Args:
//   - allocator: The recording micro allocator used during the invocation
//       process.
//   - type: Which print format should be used to output the allocation data to
//       stdout.
void LogAllocatorEvents(const tflite::RecordingMicroAllocator& allocator,
                        PrettyPrintType type);
}  // namespace tflite

#endif  // TFLM_BENCHMARK_INTERNAL_METRICS_H_
