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

#ifndef MACHINA_LITE_DELEGATES_GPU_CL_CL_PROGRAM_H_
#define MACHINA_LITE_DELEGATES_GPU_CL_CL_PROGRAM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "machina/lite/delegates/gpu/cl/cl_context.h"
#include "machina/lite/delegates/gpu/cl/cl_device.h"
#include "machina/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/compiler_options.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string CompilerOptionsToString(
    const GpuInfo& gpu_info,
    const std::vector<CompilerOptions>& compiler_options);

class CLProgram {
 public:
  CLProgram() {}
  CLProgram(cl_program program, cl_device_id device_id);

  // Move only
  CLProgram(CLProgram&& program);
  CLProgram& operator=(CLProgram&& program);
  CLProgram(const CLProgram&) = delete;
  CLProgram& operator=(const CLProgram&) = delete;

  ~CLProgram();

  cl_program program() const { return program_; }

  // Return the cl_device_id associated with the program object.
  // This can be the device associated with context on which the program object
  // has been created or can be device that was specified when a program object
  // was created using clCreateProgramWithBinary.
  cl_device_id GetDeviceId() const { return device_id_; }

  absl::Status GetBinary(std::vector<uint8_t>* result) const;

 private:
  void Release();

  cl_program program_ = nullptr;

  // reference
  cl_device_id device_id_ = nullptr;
};

absl::Status CreateCLProgram(const std::string& code,
                             const std::string& compiler_options,
                             const CLContext& context, const CLDevice& device,
                             CLProgram* result);

absl::Status CreateCLProgramFromBinary(const CLContext& context,
                                       const CLDevice& device,
                                       absl::Span<const uint8_t> binary,
                                       CLProgram* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_CL_CL_PROGRAM_H_
