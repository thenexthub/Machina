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

#ifndef MACHINA_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_
#define MACHINA_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_

#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace functor {

template <typename Device>
absl::Status DoParallelConcat(const Device& device, const Tensor& value,
                              int32_t loc, Tensor* output);

// Inplace update/add/sub values in 'y'. It computes
//   y[i, :] = v if op is I_UPDATE
//   y[i, :] += v if op is I_ADD
//   y[i, :] -= v if op is I_SUB
// Returns an error if the operation fails.
enum InplaceOpType {
  I_UPDATE,  // x = y
  I_ADD,     // x += y
  I_SUB,     // x -= y
};
template <typename Device>
absl::Status DoInplace(const Device& device, InplaceOpType op, const Tensor& i,
                       const Tensor& v, Tensor* y);
// Copies x into y.
template <typename Device>
absl::Status DoCopy(const Device& device, const Tensor& x, Tensor* y);

}  // end namespace functor
}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_
