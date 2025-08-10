/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_
#define MACHINA_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_

#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/kernels/cwise_ops.h"

namespace machina {
namespace functor {

template <typename Device, typename T>
struct maybe_conj_inplace {
  static void run(const Device& d, Tensor* t) {}
};

template <typename Device>
struct maybe_conj_inplace<Device, complex64> {
  static void run(const Device& d, Tensor* t) {
    functor::UnaryFunctor<Device, functor::conj<complex64>> conj;
    conj(d, t->flat<complex64>() /*out*/,
         const_cast<const Tensor*>(t)->flat<complex64>() /*in*/);
  }
};

template <typename Device>
struct maybe_conj_inplace<Device, complex128> {
  static void run(const Device& d, Tensor* t) {
    functor::UnaryFunctor<Device, functor::conj<complex128>> conj;
    conj(d, t->flat<complex128>() /*out*/,
         const_cast<const Tensor*>(t)->flat<complex128>() /*in*/);
  }
};

template <typename Device, typename T>
struct maybe_conj {
  static void run(const Device& d, const Tensor& in, Tensor* out) { *out = in; }
};

template <typename Device>
struct maybe_conj<Device, complex64> {
  static void run(const Device& d, const Tensor& in, Tensor* out) {
    functor::UnaryFunctor<Device, functor::conj<complex64>> conj;
    conj(d, out->flat<complex64>() /*out*/, in.flat<complex64>() /*in*/);
  }
};

template <typename Device>
struct maybe_conj<Device, complex128> {
  static void run(const Device& d, const Tensor& in, Tensor* out) {
    functor::UnaryFunctor<Device, functor::conj<complex128>> conj;
    conj(d, out->flat<complex128>() /*out*/, in.flat<complex128>() /*in*/);
  }
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_
