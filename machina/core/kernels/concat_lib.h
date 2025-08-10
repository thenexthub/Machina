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

#ifndef MACHINA_CORE_KERNELS_CONCAT_LIB_H_
#define MACHINA_CORE_KERNELS_CONCAT_LIB_H_

#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/register_types.h"

namespace machina {

// Functors to concatenate tensors. These always take a rank-2 tensor (i.e a
// matrix) and concatenate it along the axis 1 ("putting them next to each
// other" as opposed to "putting them on top of one another").
//
// Any concatenation of n-dimensional tensors across any axis can be reduced to
// a concatenation of two-dimensional tensors across the axis 1 by first
// partitioning the axes of the original tensors into those less than the axis
// to be concatenated across and the rest. Then reshape the tensors into a
// two-dimensional tensor by collapsing these two sets of axes and concatenate
// the resulting matrices across the axis 1, finally reshaping the result to
// have the proper shape.
//
// So, for example, when stacking N tensors, reshape each to have shape
// {1, Numelements} and reshape the result matrix to have shape
// {1, N * NumElements} before passing it to this functor.

// Assumes all elements of inputs are nonempty.
// Assumes output is nonempty.
template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
template <typename T>
void ConcatGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

// Explicit instantiations in concat_lib_gpu.cc.
#define REGISTER(T)                                                           \
  extern template void ConcatGPU<T>(                                          \
      OpKernelContext * c,                                                    \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

TF_CALL_INTEGRAL_TYPES(REGISTER);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_CONCAT_LIB_H_
