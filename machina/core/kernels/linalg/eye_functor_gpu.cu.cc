/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/type_traits.h"
#include "machina/core/kernels/linalg/eye_functor.h"
#include "machina/core/util/gpu_kernel_helper.h"

namespace machina {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
__global__ void EyeKernel(int num_threads, int batch_size, int m, int n,
                          Scalar* __restrict__ output_ptr) {
  const Scalar one = Scalar(1);
  const Scalar zero = Scalar(0);
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    // TODO(rmlarsen): Benchmark to see if it's just as fast to use mod (%),
    // since it's easier to read.
    const int global_row = index / n;
    const int col = index - global_row * n;
    const int batch = global_row / m;
    const int row = global_row - batch * m;
    output_ptr[index] = col == row ? one : zero;
  }
}

template <typename Scalar>
struct EyeFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::Tensor matrix_batch) {
    const int batch_size = matrix_batch.dimension(0);
    const int m = matrix_batch.dimension(1);
    const int n = matrix_batch.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
    TF_CHECK_OK(GpuLaunchKernel(EyeKernel<Scalar>, config.block_count,
                                config.thread_per_block, 0, device.stream(),
                                config.virtual_thread_count, batch_size, m, n,
                                matrix_batch.data()));
  }
};

template struct EyeFunctor<GPUDevice, float>;
template struct EyeFunctor<GPUDevice, double>;
template struct EyeFunctor<GPUDevice, complex64>;
template struct EyeFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
