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

#ifndef MACHINA_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_
#define MACHINA_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_

#include <vector>
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/numeric_op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/kernels/avgpooling_op.h"
#include "machina/core/kernels/maxpooling_op.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/platform/stream_executor.h"
#include "machina/core/util/padding.h"
#include "machina/core/util/tensor_format.h"

namespace machina {

// A helper class that launch the cudnn pooling forward operations.
template <typename T>
class DnnPoolingOp {
 public:
  typedef GPUDevice Device;
  static void Compute(OpKernelContext* context,
                      se::dnn::PoolingMode pooling_mode,
                      const std::vector<int32>& size,
                      const std::vector<int32>& stride, Padding padding,
                      std::vector<int64_t> explicit_paddings,
                      TensorFormat data_format, const Tensor& tensor_in,
                      const TensorShape& tensor_out_shape, bool propagate_nans);
};

// A helper class that launch the cudnn pooling backward operations.
// The original input and output tensors are optional for AvgPoolGrad, but
// mandatory for MaxPoolGrad.
template <typename T>
class DnnPoolingGradOp {
 public:
  typedef GPUDevice Device;
  static void Compute(OpKernelContext* context,
                      se::dnn::PoolingMode pooling_mode,
                      const std::vector<int32>& size,
                      const std::vector<int32>& stride, Padding padding,
                      std::vector<int64_t> explicit_paddings,
                      TensorFormat data_format, const Tensor* tensor_in,
                      const Tensor* tensor_out, const Tensor& out_backprop,
                      const TensorShape& tensor_in_shape, bool propagate_nans);
};

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_
