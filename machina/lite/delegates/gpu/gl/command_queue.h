/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_COMMAND_QUEUE_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_COMMAND_QUEUE_H_

#include <memory>

#include "machina/lite/delegates/gpu/common/gpu_info.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/types.h"
#include "machina/lite/delegates/gpu/gl/gl_program.h"

namespace tflite {
namespace gpu {
namespace gl {

// GL programs can be executed directly via dispatch call or using a queue
// abstraction similar to one in OpenCL and Vulkan.
// CommandQueue executes given programs in order as they come.
class CommandQueue {
 public:
  virtual ~CommandQueue() = default;

  // Dispatches a program. It may or may not call glFlush.
  virtual absl::Status Dispatch(const GlProgram& program,
                                const uint3& workgroups) = 0;

  // Called at the end of dispatching of all programs.
  virtual absl::Status Flush() = 0;

  // Waits until all programs dispatched prior this call are completed.
  virtual absl::Status WaitForCompletion() = 0;
};

// By default memory barrier is inserted after every dispatch.
std::unique_ptr<CommandQueue> NewCommandQueue(const GpuInfo& gpu_info);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_COMMAND_QUEUE_H_
