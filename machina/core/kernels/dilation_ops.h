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

#ifndef MACHINA_CORE_KERNELS_DILATION_OPS_H_
#define MACHINA_CORE_KERNELS_DILATION_OPS_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct Dilation {
  // We assume that the tensor sizes are correct.
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, typename TTypes<T, 4>::Tensor output);
};

template <typename Device, typename T>
struct DilationBackpropInput {
  // We assume that the tensor sizes are correct.
  // To avoid storing the argmax values during forward computation, we recompute
  // the argmax during backward computation, which is the reason why we provide
  // filter as argument to the backward computation routine.
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 4>::Tensor in_backprop);
};

template <typename Device, typename T>
struct DilationBackpropFilter {
  // We assume that the tensor sizes are correct.
  // To avoid storing the argmax values during forward computation, we recompute
  // the argmax during backward computation, which is the reason why we provide
  // filter as argument to the backward computation routine.
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 3>::Tensor filter_backprop);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DILATION_OPS_H_
