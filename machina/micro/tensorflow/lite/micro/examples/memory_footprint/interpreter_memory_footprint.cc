/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include <cstdint>
#include <cstdlib>

#include "machina/lite/micro/benchmarks/micro_benchmark.h"
#include "machina/lite/micro/examples/memory_footprint/models/simple_add_model_model_data.h"
#include "machina/lite/micro/micro_mutable_op_resolver.h"
#include "machina/lite/micro/micro_profiler.h"
#include "machina/lite/micro/system_setup.h"

// Use MicroBenchmarkRunner to avoid boiler plate code and more easily compare
// the size with other benchmarks such as keyword_benchmark.
using InterpreterMemoryFootprintRunner = tflite::MicroBenchmarkRunner<int16_t>;
using InterpreterMemoryFootprintOpResolver = tflite::MicroMutableOpResolver<6>;

// This binary includes the TFLM Framework (interpreter, memory planner etc),
// but without any kernels.  This is used to measure the code size of the TFLM
// Framework. This binary will produce a run time error by design since no OP is
// registered. Please see README.md for more info.
int main(int argc, char** argv) {
  // Arbitrary size.
  constexpr int kTensorArenaSize = 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
  uint8_t runner_buffer[sizeof(InterpreterMemoryFootprintRunner)];

  tflite::InitializeTarget();
  tflite::MicroProfiler profiler;

  InterpreterMemoryFootprintOpResolver op_resolver;

  // Do NOT allocate any OP so that the binary does not include any kernels.

  // Use placement new as this is the standar way to create a new object in
  // the TFLM code base. This is to avoid unnecessary dynamic memory allocation
  // code in the binary.
  // We pass an arbitrary model to the benchmark runner to ensure that the TFLM
  // framework can successfully go through all the steps needed to perform the
  // initialization and memory planning needed prior to running inference on a
  // model. The specifics of model itself (size, ops ...) are not important
  // since we do not actually run any inference.
  InterpreterMemoryFootprintRunner* runner = new (runner_buffer)
      InterpreterMemoryFootprintRunner(g_simple_add_model_model_data,
                                       &op_resolver, tensor_arena,
                                       kTensorArenaSize, &profiler);

  runner->RunSingleIteration();
}
