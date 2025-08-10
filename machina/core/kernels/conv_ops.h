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

#ifndef MACHINA_CORE_KERNELS_CONV_OPS_H_
#define MACHINA_CORE_KERNELS_CONV_OPS_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/platform/mem.h"
#include "machina/core/util/tensor_format.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/kernels/conv_ops_gpu.h"
#include "machina/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

namespace machina {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchConv2DOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format);
};

template <typename Device, typename T>
struct LaunchConvOp {
  void operator()(OpKernelContext* context, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter,
                  const std::vector<int64>& dilations,
                  const std::vector<int64>& strides, Padding padding,
                  const std::vector<int64_t>& explicit_paddings,
                  TensorFormat data_format, Tensor* output);
};

#if GOOGLE_CUDA || MACHINA_USE_ROCM
template <typename T>
struct LaunchConv2DOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format);
};

template <typename T>
struct LaunchConvOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* context, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter,
                  const std::vector<int64>& dilations,
                  const std::vector<int64>& strides, const Padding padding,
                  const std::vector<int64_t>& explicit_paddings,
                  TensorFormat data_format, Tensor* output);
};
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

// Used to keep track of persistent memory buffers used within the op.
// It uses malloc and free to avoid the time cost of initializing the memory.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(port::Malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { port::Free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() const { return "Im2ColBufferResource"; }
};

// Convolution parameters specified by Op attributes.
struct Conv2DParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
  std::vector<int64_t> explicit_paddings;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64_t out_rows;
  int64_t out_cols;
  int64_t pad_rows_before;
  int64_t pad_rows_after;
  int64_t pad_cols_before;
  int64_t pad_cols_after;
};

// Initializes and validates Conv2D parameters configured by OpKernel
// attributes.
absl::Status InitConv2DParameters(const OpKernelConstruction* context,
                                  Conv2DParameters* params);

// Computes and validates convolutions dimensions from Conv2D parameters. If
// parameters are valid, dimensions will be updated with derived convolution
// dimensions, otherwise an error will be returned.
absl::Status ComputeConv2DDimension(const Conv2DParameters& params,
                                    const Tensor& input, const Tensor& filter,
                                    Conv2DDimensions* dimensions);

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_CONV_OPS_H_
