/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_LITE_PROFILING_MEMORY_LATENCY_LOGGER_H_
#define MACHINA_LITE_PROFILING_MEMORY_LATENCY_LOGGER_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "machina/lite/profiling/memory_usage_monitor.h"

namespace tflite {
namespace profiling {
namespace memory {

// This class is used to measure the memory and latency of the surrounding code
// block. Example usage:
//   MemoryLatencyLogger logger;
//   logger.Start();
//     Code block
//   logger.Stop("Code block");

// This class is thread-unsafe.
class MemoryLatencyLogger {
 public:
  MemoryLatencyLogger();
  // Starts the memory and latency monitoring.
  void Start();
  // Stops the memory and latency monitoring and logs the results.
  void Stop(absl::string_view log_message);

 private:
  // The memory usage monitor.
  std::unique_ptr<MemoryUsageMonitor> mem_monitor_;
  // The start time of the memory and latency monitoring.
  absl::Time start_;
};

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

#endif  // MACHINA_LITE_PROFILING_MEMORY_LATENCY_LOGGER_H_
