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

#ifndef MACHINA_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_
#define MACHINA_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_

#include <string>

#include "machina/lite/delegates/gpu/cl/util.h"
#include "machina/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// @return if error_code is success, then return OK status. Otherwise translates
// error code into a message.
inline absl::Status GetOpenCLError(cl_int error_code) {
  if (error_code == CL_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError("OpenCL error: " +
                             CLErrorCodeToString(error_code));
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_
