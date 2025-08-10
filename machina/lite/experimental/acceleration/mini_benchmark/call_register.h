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
#ifndef MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_CALL_REGISTER_H_
#define MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_CALL_REGISTER_H_
#include "machina/lite/core/c/common.h"
namespace tflite {
namespace acceleration {
namespace ops {
// CALL op can be used to invoke a subgraph a given number of times.
TfLiteRegistration* Register_CALL();

typedef struct {
  // Index of the subgraph that needs to be invoked.
  // Subgraph should have batch size 1.
  int subgraph_index;
  // The number of times the CALL op should call the subgraph.
  // The inputs to the call op are expected to have this value as their batch
  // size.
  int loop_count;
} TfLiteCallParams;

}  // namespace ops
}  // namespace acceleration
}  // namespace tflite
#endif  // MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_CALL_REGISTER_H_
