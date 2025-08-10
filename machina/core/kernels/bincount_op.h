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

#ifndef MACHINA_CORE_KERNELS_BINCOUNT_OP_H_
#define MACHINA_CORE_KERNELS_BINCOUNT_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/errors.h"

namespace machina {

namespace functor {

template <typename Device, typename Tidx, typename T, bool binary_count>
struct BincountFunctor {
  static absl::Status Compute(OpKernelContext* context,
                              const typename TTypes<Tidx, 1>::ConstTensor& arr,
                              const typename TTypes<T, 1>::ConstTensor& weights,
                              typename TTypes<T, 1>::Tensor& output,
                              const Tidx num_bins);
};

template <typename Device, typename Tidx, typename T, bool binary_count>
struct BincountReduceFunctor {
  static absl::Status Compute(OpKernelContext* context,
                              const typename TTypes<Tidx, 2>::ConstTensor& in,
                              const typename TTypes<T, 2>::ConstTensor& weights,
                              typename TTypes<T, 2>::Tensor& out,
                              const Tidx num_bins);
};

}  // end namespace functor

}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_BINCOUNT_OP_H_
