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

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define EIGEN_USE_GPU

#include <assert.h>
#include <stdio.h>

#include "machina/core/kernels/random_op_gpu.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor_types.h"
#include "machina/core/lib/random/philox_random.h"
#include "machina/core/lib/random/random_distributions.h"

namespace machina {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Explicit instantiation of the GPU distributions functors
// clang-format off
// NVCC cannot handle ">>" properly
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, uint32> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, uint64> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, double> >;
// clang-format on

}  // namespace functor
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
