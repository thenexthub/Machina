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

#ifndef MACHINA_LITE_DELEGATES_GPU_CL_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_CL_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "machina/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tensor.h"
#include "machina/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string CLErrorCodeToString(cl_int error_code);

int ChannelTypeToSizeInBytes(cl_channel_type type);

bool OpenCLSupported();

template <DataType S, typename T>
void CopyLinearFLT4(const tflite::gpu::Tensor<Linear, S>& src,
                    absl::Span<T> dst) {
  const int dst_depth = dst.size();
  for (int d = 0; d < dst_depth; ++d) {
    T val;
    for (int i = 0; i < 4; ++i) {
      const int dst_ch = d * 4 + i;
      val[i] = dst_ch >= src.shape.v ? 0.0f : src.data[dst_ch];
    }
    dst[d] = val;
  }
}

absl::Status CreateCLBuffer(cl_context context, int size_in_bytes,
                            bool read_only, void* data, cl_mem* result);

absl::Status CreateCLSubBuffer(cl_context context, cl_mem parent,
                               size_t origin_in_bytes, size_t size_in_bytes,
                               bool read_only, cl_mem* result);

absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void* data,
                               cl_mem* result);

absl::Status CreateQcomConvolutionFilter(cl_context context, int kernel_x,
                                         int kernel_y, cl_mem* filter,
                                         const void* data);

std::vector<std::pair<std::string, std::string>> GetClSpecificDefines();

// Vendor extensions that cannot be used in open-source
std::vector<std::string> GetUnsupportedExtensions();

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_CL_UTIL_H_
