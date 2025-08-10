/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_LITE_MICRO_MICRO_PROFILER_INTERFACE_H_
#define MACHINA_LITE_MICRO_MICRO_PROFILER_INTERFACE_H_

#include <cstdint>

namespace tflite {

// Interface class that the TFLM framework relies on for profiling.
class MicroProfilerInterface {
 public:
  virtual ~MicroProfilerInterface() {}

  // Marks the start of a new event and returns an event handle that can be used
  // to mark the end of the event via EndEvent.
  virtual uint32_t BeginEvent(const char* tag) = 0;

  // Marks the end of an event associated with event_handle.
  virtual void EndEvent(uint32_t event_handle) = 0;
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_MICRO_PROFILER_INTERFACE_H_
