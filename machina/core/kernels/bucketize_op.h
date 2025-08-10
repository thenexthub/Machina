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

#ifndef MACHINA_CORE_KERNELS_BUCKETIZE_OP_H_
#define MACHINA_CORE_KERNELS_BUCKETIZE_OP_H_

#include <vector>
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/errors.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct BucketizeFunctor {
  static absl::Status Compute(OpKernelContext* context,
                              const typename TTypes<T, 1>::ConstTensor& input,
                              const std::vector<float>& boundaries_vector,
                              typename TTypes<int32, 1>::Tensor& output);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_BUCKETIZE_OP_H_
