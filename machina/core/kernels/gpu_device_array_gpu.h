/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

// Contains structs and functions to be included in device code.

#ifndef MACHINA_CORE_KERNELS_GPU_DEVICE_ARRAY_GPU_H_
#define MACHINA_CORE_KERNELS_GPU_DEVICE_ARRAY_GPU_H_

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)

namespace machina {

// To decode on the device side, use GetGpuDeviceArrayOnDevice.
// To encode on the host side, use GpuDeviceArrayOnHost.
template <typename ValueType, int MaxInlineValues = 8>
struct GpuDeviceArrayStruct {
  int32 size;
  // used if size <= MaxInlineValues;
  ValueType inline_values[MaxInlineValues];
  ValueType* out_of_line_values = nullptr;  // used if size > MaxInlineValues;
};

template <typename ValueType, int MaxInlineValues = 8>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ValueType* GetGpuDeviceArrayOnDevice(
    GpuDeviceArrayStruct<ValueType, MaxInlineValues>* data) {
  if (data->size <= MaxInlineValues) {
    return data->inline_values;
  } else {
    return data->out_of_line_values;
  }
}

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

#endif  // MACHINA_CORE_KERNELS_GPU_DEVICE_ARRAY_GPU_H_
