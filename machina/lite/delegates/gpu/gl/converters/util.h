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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/types.h"
#include "machina/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {

inline std::string GetShaderHeader(const uint3& localsize) {
  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

inline uint32_t BytesForPHWC4(const BHWC& shape) {
  return shape.b * shape.h * shape.w * AlignByN(shape.c, 4) * sizeof(float);
}

inline uint32_t BytesForBHWC(const BHWC& shape) {
  return shape.DimensionsProduct() * sizeof(float);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_
