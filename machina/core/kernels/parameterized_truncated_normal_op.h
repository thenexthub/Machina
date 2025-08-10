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

#ifndef MACHINA_CORE_KERNELS_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_
#define MACHINA_CORE_KERNELS_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_

#include "machina/core/framework/tensor_types.h"
#include "machina/core/lib/random/random_distributions.h"
#include "machina/core/util/bcast.h"

namespace machina {

class OpKernelContext;

namespace functor {

// Sample a truncated normal random variable, with mean, stddev, minval, and
// maxval parameters for each batch. Uses two rejection sampling algorithms
// described in http://rd.springer.com/article/10.1007/BF00143942 and a randn
// rejection sampler when most of the normal is inside the bounds.
//
// Either minval may be -infinity, or maxval may be +infinity. If the interval
// (minval, maxval) is empty, the result is NaN.
template <typename Device, typename T>
struct TruncatedNormalFunctor {
  void operator()(OpKernelContext* ctx, const Device& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output);
};

// This version supports broadcasting of the arguments, as well as puts
// the sample dimension on the left.
template <typename Device, typename T>
struct TruncatedNormalFunctorV2 {
  void operator()(OpKernelContext* ctx, const Device& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  const BCastList<4>& bcast,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output);
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_PARAMETERIZED_TRUNCATED_NORMAL_OP_H_
