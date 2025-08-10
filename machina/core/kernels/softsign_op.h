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

#ifndef MACHINA_CORE_KERNELS_SOFTSIGN_OP_H_
#define MACHINA_CORE_KERNELS_SOFTSIGN_OP_H_
// Functor definition for SoftsignOp and SoftsignGradOp, must be compilable by
// nvcc.

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor_types.h"

namespace machina {
namespace functor {

// Functor used by SoftsignOp to do the computations.
template <typename Device, typename T>
struct Softsign {
  // Computes Softsign activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) =
        features / (features.abs() + features.constant(T(1)));
  }
};

// Functor used by SoftsignGradOp to do the computations.
template <typename Device, typename T>
struct SoftsignGrad {
  // Computes SoftsignGrad backprops.
  //
  // gradients: gradients backpropagated to the Softsign op.
  // features: inputs that were passed to the Softsign op.
  // backprops: gradients to backpropagate to the Softsign inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        gradients / (features.abs() + features.constant(T(1))).square();
  }
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_SOFTSIGN_OP_H_
