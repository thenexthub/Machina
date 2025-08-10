/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_H_
#define MACHINA_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_H_

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/rng_alg.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina {

inline absl::Status CheckKeyCounterShape(int minimum_counter_size,
                                         TensorShape const& key_shape,
                                         TensorShape const& counter_shape) {
  if (!(key_shape.dims() == 1 && key_shape.dim_size(0) == RNG_KEY_SIZE)) {
    return errors::InvalidArgument(
        "key must have shape [", RNG_KEY_SIZE, "], not ",
        key_shape.DebugString(),
        ". (Note that batched keys are not supported yet.)");
  }
  if (!(counter_shape.dims() == 1 &&
        counter_shape.dim_size(0) >= minimum_counter_size)) {
    return errors::InvalidArgument(
        "counter must be a vector with length at least ", minimum_counter_size,
        "; got shape: ", counter_shape.DebugString(),
        ". (Note that batched counters are not supported yet.)");
  }
  return absl::OkStatus();
}

// A base class for kernels of stateless RNG ops that take shape, key, counter
// and algorithm as the first 4 inputs.
class StatelessRandomOpBaseWithKeyCounter : public OpKernel {
 public:
  explicit StatelessRandomOpBaseWithKeyCounter(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:
  // The part of Compute that depends on device, type, and distribution.
  // Must be a tail call because it doesn't report error via return value.
  virtual void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                    const Tensor& counter, Tensor* output) = 0;
};

}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_H_
