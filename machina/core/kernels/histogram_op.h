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

#ifndef MACHINA_CORE_KERNELS_HISTOGRAM_OP_H_
#define MACHINA_CORE_KERNELS_HISTOGRAM_OP_H_

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/errors.h"

namespace machina {
namespace functor {

template <typename Device, typename T, typename Tout>
struct HistogramFixedWidthFunctor {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& values,
      const typename TTypes<T, 1>::ConstTensor& value_range, int32_t nbins,
      typename TTypes<Tout, 1>::Tensor& out);
};

}  // end namespace functor
}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_HISTOGRAM_OP_H_
