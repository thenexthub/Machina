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

#ifndef MACHINA_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_
#define MACHINA_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_

#include "machina/lite/micro/micro_interpreter.h"
#include "machina/lite/micro/micro_profiler_interface.h"
#include "machina/lite/micro/recording_micro_allocator.h"

namespace tflite {

// Utility subclass that enables internal recordings of the MicroInterpreter.
// This class should be used to audit and analyze memory arena usage for a given
// model and interpreter.
//
// After construction and the first Invoke() or AllocateTensors() call - the
// memory usage is recorded and available through the GetMicroAllocator()
// function. See RecordingMicroAlloctor for more details on what is currently
// recorded from arena allocations.
//
// It is recommended for users to increase the tensor arena size by at least 1kb
// to ensure enough additional memory is available for internal recordings.
class RecordingMicroInterpreter : public MicroInterpreter {
 public:
  RecordingMicroInterpreter(const Model* model,
                            const MicroOpResolver& op_resolver,
                            uint8_t* tensor_arena, size_t tensor_arena_size,
                            MicroResourceVariables* resource_variable = nullptr,
                            MicroProfilerInterface* profiler = nullptr)
      : MicroInterpreter(
            model, op_resolver,
            RecordingMicroAllocator::Create(tensor_arena, tensor_arena_size),
            resource_variable, profiler),
        recording_micro_allocator_(
            static_cast<const RecordingMicroAllocator&>(allocator())) {}

  RecordingMicroInterpreter(const Model* model,
                            const MicroOpResolver& op_resolver,
                            RecordingMicroAllocator* allocator,
                            MicroResourceVariables* resource_variable = nullptr,
                            MicroProfilerInterface* profiler = nullptr)
      : MicroInterpreter(model, op_resolver, allocator, resource_variable,
                         profiler),
        recording_micro_allocator_(*allocator) {}

  const RecordingMicroAllocator& GetMicroAllocator() const {
    return recording_micro_allocator_;
  }

 private:
  const RecordingMicroAllocator& recording_micro_allocator_;
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_
