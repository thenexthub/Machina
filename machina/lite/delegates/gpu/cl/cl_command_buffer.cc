/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/lite/delegates/gpu/cl/cl_command_buffer.h"

#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "machina/lite/delegates/gpu/cl/cl_command_queue.h"
#include "machina/lite/delegates/gpu/cl/cl_event.h"
#include "machina/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "machina/lite/delegates/gpu/cl/util.h"
#include "machina/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

CLCommandBuffer::CLCommandBuffer(CLCommandBuffer&& cb) : cb_(cb.cb_) {
  cb.cb_ = nullptr;
}

CLCommandBuffer& CLCommandBuffer::operator=(CLCommandBuffer&& cb) {
  if (this != &cb) {
    Release();
    std::swap(cb_, cb.cb_);
  }
  return *this;
}

void CLCommandBuffer::Release() {
  if (cb_) {
    clReleaseCommandBufferKHR(cb_);
    cb_ = nullptr;
  }
}

absl::Status CLCommandBuffer::Init(CLCommandQueue* queue,
                                   bool simultaneous_use) {
  cl_int errcode_ret = CL_SUCCESS;
  std::vector<cl_command_buffer_properties_khr> properties;
  if (simultaneous_use) {
    properties.push_back(CL_COMMAND_BUFFER_FLAGS_KHR);
    properties.push_back(CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR);
  }
  properties.push_back(0);
  cl_command_buffer_properties_khr* properties_ptr =
      properties.size() != 1 ? properties.data() : nullptr;
  cl_command_queue cmd_queue = queue->queue();
  cb_ = clCreateCommandBufferKHR(1, &cmd_queue, properties_ptr, &errcode_ret);
  if (errcode_ret != CL_SUCCESS) {
    return absl::InternalError(absl::StrCat("Failed clCreateCommandBufferKHR.",
                                            CLErrorCodeToString(errcode_ret)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandBuffer::Finalize() {
  cl_int errcode_ret = clFinalizeCommandBufferKHR(cb_);
  if (errcode_ret != CL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Failed clFinalizeCommandBufferKHR.",
                     CLErrorCodeToString(errcode_ret)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandBuffer::Enqueue(CLCommandQueue* queue, CLEvent* event) {
  cl_event resulting_event;
  cl_command_queue cmd_queue = queue->queue();
  cl_int errcode_ret = clEnqueueCommandBufferKHR(
      1, &cmd_queue, cb_, 0, nullptr, event ? &resulting_event : nullptr);
  if (errcode_ret != CL_SUCCESS) {
    return absl::InternalError(absl::StrCat("Failed clEnqueueCommandBufferKHR.",
                                            CLErrorCodeToString(errcode_ret)));
  }
  if (event) {
    *event = CLEvent(resulting_event);
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
