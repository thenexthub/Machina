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
#ifndef MACHINA_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_
#define MACHINA_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_

#include <functional>
#include <vector>

#include "machina/lite/core/api/profiler.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/interpreter.h"

namespace tflite::profiling {

// The SubgraphTensorProfiler is invoked for every tensor in a subgraph at the
// end of the subgraph's execution. This profiler is constructed with a user
// provided callback to run on each tensor in the subgraph.
class SubgraphTensorProfiler : public tflite::Profiler {
 public:
  using CallbackT = std::function<void(const TfLiteTensor*)>;

  SubgraphTensorProfiler(const Interpreter& interpreter, CallbackT callback);

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  void EndEvent(uint32_t event_handle) override;

 private:
  // A mapping between event IDs and the subgraph that owns the event ID.
  std::vector<int64_t> events_;

  // A handle to the active TFLite interpreter.
  const Interpreter& interpreter_;

  // A user provided callback to run on each tensor in the subgraph. The
  // callback signature is:
  //
  //  void Callback(const TfLiteTensor* tensor);
  CallbackT callback_;
};

}  // namespace tflite::profiling

#endif  // MACHINA_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_
