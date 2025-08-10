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

// Helper functions to run 3d pooling on GPU using CuDNN.

#ifndef MACHINA_CORE_KERNELS_CUDNN_POOLING_GPU_H_
#define MACHINA_CORE_KERNELS_CUDNN_POOLING_GPU_H_

#include <array>

#include "machina/core/framework/op_kernel.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/platform/stream_executor.h"
#endif

#include "machina/core/util/padding.h"

namespace machina {

#if GOOGLE_CUDA || MACHINA_USE_ROCM

// Runs (avg/max)pooling on GPU.
// Dimension order for all array arguments is: x, y, z.
template <typename T>
class DnnPooling3dOp {
 public:
  static void Compute(OpKernelContext* context,
                      se::dnn::PoolingMode pooling_mode,
                      const std::array<int64, 3>& size,
                      const std::array<int64, 3>& stride,
                      const std::array<int64, 3>& padding,
                      TensorFormat data_format, const Tensor& tensor_in,
                      Tensor* output);
};

// Computes the gradient of (avg/max)pooling on GPU.
// Dimension order for all array arguments is: x, y, z.
template <typename T>
class DnnPooling3dGradOp {
 public:
  static void Compute(OpKernelContext* context,
                      se::dnn::PoolingMode pooling_mode,
                      const std::array<int64, 3>& window,
                      const std::array<int64, 3>& stride,
                      const std::array<int64, 3>& padding,
                      const std::array<int64, 3>& output_size,
                      TensorFormat data_format, const Tensor& out_backprop,
                      const TensorShape& tensor_in_shape,
                      const Tensor* tensor_in, const Tensor* tensor_out,
                      Tensor* input_backprop);
};

#endif

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_CUDNN_POOLING_GPU_H_
