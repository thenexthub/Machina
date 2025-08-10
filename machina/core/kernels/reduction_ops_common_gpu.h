/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_
#define MACHINA_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_

#if !GOOGLE_CUDA && !MACHINA_USE_ROCM
#error This file must only be included when building with GPU support
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"

namespace machina {
namespace functor {

template <typename Reducer>
struct ReduceFunctor<Eigen::GpuDevice, Reducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);

  template <typename OUT_T>
  static void FillIdentity(const Eigen::GpuDevice& d, OUT_T out,
                           const Reducer& reducer);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_
