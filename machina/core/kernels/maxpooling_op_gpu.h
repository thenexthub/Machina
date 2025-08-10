/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#if !GOOGLE_CUDA && !MACHINA_USE_ROCM
#error This file must only be included when building with Cuda or ROCm support
#endif

#ifndef MACHINA_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
#define MACHINA_CORE_KERNELS_MAXPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/tensor_format.h"

namespace machina {

namespace functor {
// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
template <typename T>
struct MaxPoolForwardWithOptionalArgmax {
  bool operator()(const T* bottom_data, const int batch, const int height,
                  const int width, const int channels, const int pooled_height,
                  const int pooled_width, const int kernel_h,
                  const int kernel_w, const int stride_h, const int stride_w,
                  const int pad_t, const int pad_l, T* top_data, int64_t* mask,
                  const Eigen::GpuDevice& d, bool propagate_nans,
                  const bool include_batch_in_index);
};

struct MaxPoolForwardNoMask_NCHW_VECT_C {
  bool operator()(const int32* bottom_data, const int batch, const int height,
                  const int width, int channels, const int pooled_height,
                  const int pooled_width, const int kernel_h,
                  const int kernel_w, const int stride_h, const int stride_w,
                  const int pad_t, const int pad_l, int32* top_data,
                  const Eigen::GpuDevice& d);
};

template <typename T>
struct MaxPoolBackwardWithArgmax {
  bool operator()(const int output_size, const int input_size,
                  const T* top_diff, const int64_t* mask, const int top_offset,
                  const int bottom_offset, T* bottom_diff,
                  const Eigen::GpuDevice& d, const bool include_batch_in_index);
};

template <typename T>
struct MaxPoolGradBackwardWithArgmax {
  bool operator()(const int output_size, const int input_size,
                  const T* top_diff, const int64_t* mask, const int top_offset,
                  const int bottom_offset, T* bottom_diff,
                  const Eigen::GpuDevice& d, const bool include_batch_in_index);
};

template <typename T>
struct MaxPoolGradBackwardNoMask {
  bool operator()(TensorFormat data_format, const T* bottom_data,
                  const T* output_data, const int batch,
                  const int pooled_height, const int pooled_width,
                  const int channels, const int height, const int width,
                  const int kernel_h, const int kernel_w, const int stride_h,
                  const int stride_w, const int pad_t, const int pad_l,
                  const T* top_diff, T* bottom_diff, const Eigen::GpuDevice& d);
};

}  // namespace functor

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
