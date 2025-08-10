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

#ifndef MACHINA_CORE_KERNELS_RANDOM_OP_H_
#define MACHINA_CORE_KERNELS_RANDOM_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/lib/random/random_distributions.h"

namespace machina {

class OpKernelContext;

namespace functor {

template <typename Device, class Distribution>
struct FillPhiloxRandom;

typedef Eigen::ThreadPoolDevice CPUDevice;
// Declares the partially CPU-specialized functor struct.
//
// NOTE: Due to inlining done by the compiler, you may need to add
// explicit instantiation of the functor in random_op.cc.  See example
// functor::FillPhiloxRandom<CPUDevice, random::UniformDistribution>.
//
// This functor can take the PhiloxRandom input from either device memory `key`
// and `counter` or a stack value `gen`. If both `key` and `counter` are not
// nullptr, they provide the input; otherwise `gen` provides the input.
template <class Distribution>
struct FillPhiloxRandom<CPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, const uint64* key,
                  const uint64* counter, random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64_t size,
                  Distribution dist);
};

#if GOOGLE_CUDA || MACHINA_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
// Declares the partially GPU-specialized functor struct.
template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d, const uint64* key,
                  const uint64* counter, random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64_t size,
                  Distribution dist);
};
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_RANDOM_OP_H_
