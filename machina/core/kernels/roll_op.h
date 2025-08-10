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

#ifndef MACHINA_CORE_KERNELS_ROLL_OP_H_
#define MACHINA_CORE_KERNELS_ROLL_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/numeric_types.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct Roll {
  // dim_size - the size of each dimension
  // dim_range - the number of indices over in the flattened tensor
  //    you need to skip in order to make it over from one side of a dimension
  //    to the other. Used to make the shifts wrap around after a threshold.
  // threshold - the index for each dimension that the roll starts to wrap
  //    back to the front
  // isd - inner shift dimension
  void operator()(const OpKernelContext* context, const int64_t num_elements,
                  const int num_dims, const absl::Span<const int32> dim_size,
                  const T* input, T* output,
                  const absl::Span<const int32> threshold,
                  const absl::Span<const int64_t> dim_range, const int64_t isd);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_ROLL_OP_H_
