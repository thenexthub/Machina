/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define EIGEN_USE_GPU

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/kernels/searchsorted_op.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/gpu_kernel_helper.h"

namespace machina {
typedef Eigen::GpuDevice GPUDevice;

namespace {
template <typename T, typename OutType>
__global__ void UpperBoundKernel(const T* __restrict__ sorted_inputs,
                                 int batch_size, int sorted_inputs_size,
                                 int values_size, const T* __restrict__ values,
                                 OutType* __restrict__ outputs) {
  GPU_1D_KERNEL_LOOP(work_unit_id, values_size * batch_size) {
    int bid = work_unit_id / values_size;
    T value = values[work_unit_id];
    outputs[work_unit_id] = gpu_helper::upper_bound<T, OutType>(
        sorted_inputs + bid * sorted_inputs_size, sorted_inputs_size, value);
  }
}

template <typename T, typename OutType>
__global__ void LowerBoundKernel(const T* __restrict__ sorted_inputs,
                                 int batch_size, int sorted_inputs_size,
                                 int values_size, const T* __restrict__ values,
                                 OutType* __restrict__ outputs) {
  GPU_1D_KERNEL_LOOP(work_unit_id, values_size * batch_size) {
    int bid = work_unit_id / values_size;
    T value = values[work_unit_id];
    outputs[work_unit_id] = gpu_helper::lower_bound<T, OutType>(
        sorted_inputs + bid * sorted_inputs_size, sorted_inputs_size, value);
  }
}
}  // namespace

namespace functor {
template <typename T, typename OutType>
struct UpperBoundFunctor<GPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (values.size() == 0) {
      // GetGpuLaunchConfig requires work_element_count > 0
      return OkStatus();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(values.size(), device);

    TF_CHECK_OK(GpuLaunchKernel(
        UpperBoundKernel<T, OutType>, config.block_count,
        config.thread_per_block, 0, device.stream(), sorted_inputs.data(),
        batch_size, num_inputs, num_values, values.data(), output->data()));

    return OkStatus();
  }
};

template <typename T, typename OutType>
struct LowerBoundFunctor<GPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (values.size() == 0) {
      // GetGpuLaunchConfig requires work_element_count > 0
      return OkStatus();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(values.size(), device);

    TF_CHECK_OK(GpuLaunchKernel(
        LowerBoundKernel<T, OutType>, config.block_count,
        config.thread_per_block, 0, device.stream(), sorted_inputs.data(),
        batch_size, num_inputs, num_values, values.data(), output->data()));

    return OkStatus();
  }
};
}  // namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::UpperBoundFunctor<GPUDevice, type, int32>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::UpperBoundFunctor<GPUDevice, type, int64>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::LowerBoundFunctor<GPUDevice, type, int32>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

#define REGISTER_GPU_SPEC(type) \
  template struct functor::LowerBoundFunctor<GPUDevice, type, int64>;

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
