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

#ifndef MACHINA_LITE_DELEGATES_GPU_ASYNC_BUFFERS_H_
#define MACHINA_LITE_DELEGATES_GPU_ASYNC_BUFFERS_H_

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include <GLES3/gl31.h>

#include "absl/status/status.h"
#include "machina/lite/delegates/gpu/api.h"
#include "machina/lite/delegates/gpu/common/data_type.h"

extern "C" typedef struct AHardwareBuffer AHardwareBuffer;

namespace tflite {
namespace gpu {

class AsyncBuffer {
 private:
  int bytes_;                                // Number of bytes in the buffer
  bool valid_ = false;                       // Have we mapped to SSBO already
  GLuint opengl_buffer_ = GL_INVALID_INDEX;  // SSBO buffer id
  AHardwareBuffer* ahwb_ = nullptr;

  // Where the AHWB<->SSBO mapping occurs
  absl::Status MapAHardwareBufferToGlBuffer();
  // Allocate SSBO, call the AHWB<->SSBO mapping; fail gracefully if needed.
  absl::Status AllocateOpenGlBuffer();

 public:
  explicit AsyncBuffer(TensorObjectDef tensor_def, AHardwareBuffer* ahwb) {
    bytes_ = NumElements(tensor_def) * SizeOf(tensor_def.object_def.data_type);
    ahwb_ = ahwb;
  }
  // Map the AHWB (from class constructor) to an SSBO id
  absl::Status GetOpenGlBuffer(GLuint& buffer_ref);
};

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_ASYNC_BUFFERS_H_
