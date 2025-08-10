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

#ifndef MACHINA_CORE_KERNELS_SPLIT_LIB_GPU_H_
#define MACHINA_CORE_KERNELS_SPLIT_LIB_GPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/gpu_device_array_gpu.h"
#include "machina/core/kernels/split_lib.h"

namespace machina {

template <typename T>
struct SplitOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, const T* input, int32_t prefix_dim_size,
           int32_t split_dim_size, int32_t suffix_dim_size,
           const GpuDeviceArrayStruct<T*>& output_ptr_data);
};

template <typename T, typename IntType>
struct SplitVOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, bool fixed, const T* input,
           int total_cols, int total_rows,
           const GpuDeviceArrayStruct<IntType>& output_scan,
           const GpuDeviceArrayStruct<T*>& output_ptr_data);
};

// Explicit instantiations in split_lib_gpu.cu.cc.
#define REGISTER_GPU_KERNEL(T)                        \
  extern template struct SplitOpGPULaunch<T>;         \
  extern template struct SplitVOpGPULaunch<T, int8>;  \
  extern template struct SplitVOpGPULaunch<T, int32>; \
  extern template struct SplitVOpGPULaunch<T, int64_t>;

TF_CALL_uint8(REGISTER_GPU_KERNEL);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_SPLIT_LIB_GPU_H_
