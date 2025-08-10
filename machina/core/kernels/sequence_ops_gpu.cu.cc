/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/kernels/sequence_ops.h"
#include "machina/core/util/gpu_kernel_helper.h"

namespace machina {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
__global__ void RangeKernel(int64_t size, T start, T delta,
                            T* __restrict__ output) {
  for (int64_t i : GpuGridRangeX(size)) {
    output[i] = start + static_cast<T>(i) * delta;
  }
}

}  // namespace

namespace functor {

template <typename T>
struct RangeFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, int64_t size, T start, T delta,
                  typename TTypes<T>::Flat output) const {
    const GPUDevice& device = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(
        size, device, &RangeKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(context,
                   GpuLaunchKernel(RangeKernel<T>, config.block_count,
                                   config.thread_per_block, 0, device.stream(),
                                   size, start, delta, output.data()));
  }
};

}  // namespace functor

#define DEFINE_FUNCTOR(T) template struct functor::RangeFunctor<GPUDevice, T>;
TF_CALL_half(DEFINE_FUNCTOR);
TF_CALL_bfloat16(DEFINE_FUNCTOR);
TF_CALL_float(DEFINE_FUNCTOR);
TF_CALL_double(DEFINE_FUNCTOR);
TF_CALL_int32(DEFINE_FUNCTOR);
TF_CALL_int64(DEFINE_FUNCTOR);
#undef DEFINE_FUNCTOR

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
