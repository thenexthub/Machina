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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include "machina/core/kernels/image/colorspace_op.h"

#include <algorithm>
#include <cmath>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class RGBToHSVOp : public OpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    // Make a canonical image, maintaining the last (channel) dimension, while
    // flattening all others do give the functor easy to work with data.
    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();

    Tensor trange;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({input_data.dimension(0)}),
                                        &trange));

    typename TTypes<T, 1>::Tensor range(trange.tensor<T, 1>());

    functor::RGBToHSV<Device, T>()(context->eigen_device<Device>(), input_data,
                                   range, output_data);
  }
};

template <typename Device, typename T>
class HSVToRGBOp : public OpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();

    functor::HSVToRGB<Device, T>()(context->eigen_device<Device>(), input_data,
                                   output_data);
  }
};

#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RGBToHSV").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RGBToHSVOp<CPUDevice, T>);                                  \
  template class RGBToHSVOp<CPUDevice, T>;                        \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("HSVToRGB").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      HSVToRGBOp<CPUDevice, T>);                                  \
  template class HSVToRGBOp<CPUDevice, T>;
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU(T)                                               \
  template <>                                                        \
  void RGBToHSV<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 1>::Tensor range, TTypes<T, 2>::Tensor output_data); \
  extern template struct RGBToHSV<GPUDevice, T>;                     \
  template <>                                                        \
  void HSVToRGB<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 2>::Tensor output_data);                             \
  extern template struct HSVToRGB<GPUDevice, T>;
TF_CALL_float(DECLARE_GPU);
TF_CALL_double(DECLARE_GPU);
}  // namespace functor
#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RGBToHSV").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RGBToHSVOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("HSVToRGB").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      HSVToRGBOp<GPUDevice, T>);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#endif

}  // namespace machina
