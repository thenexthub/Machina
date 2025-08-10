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

#include <cstddef>

#include "machina/lite/core/macros.h"
#include "machina/lite/machina_profiler_logger.h"

// Use weak symbols here (even though they are guarded by macros) to avoid
// build breakage when building a benchmark requires TFLite runs. The main
// benchmark library should have tensor_profiler_logger dependency.
// Strong symbol definitions can be found in machina_profiler_logger.cc.

namespace tflite {

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpPrepare(const char* op_name,
                                             int subgraph_index,
                                             int node_index) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK machina::profiler::TraceMe* OnTfLiteSubgraphInvoke(
    const char* subgraph_name, int subgraph_index) {
  return nullptr;
}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteSubgraphInvokeEnd(
    machina::profiler::TraceMe* trace_me) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK machina::profiler::TraceMe* OnTfLiteOpInvoke(
    const char* op_name, int subgraph_index, int node_index) {
  return nullptr;
}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpInvokeEnd(
    machina::profiler::TraceMe* trace_me) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorAlloc(TfLiteTensor* tensor,
                                               size_t num_bytes) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorDealloc(TfLiteTensor* tensor) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaAlloc(int subgraph_index, int arena_id,
                                              size_t num_bytes) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaDealloc(int subgraph_index,
                                                int arena_id,
                                                size_t num_bytes) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void PauseHeapMonitoring(bool pause) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// machina_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteInterpreterEnd() {}

}  // namespace tflite
