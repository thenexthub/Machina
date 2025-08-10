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

#ifndef MACHINA_CORE_KERNELS_WHERE_OP_H_
#define MACHINA_CORE_KERNELS_WHERE_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {

#define TF_CALL_WHERE_GPU_TYPES(m) \
  TF_CALL_int8(m);                 \
  TF_CALL_uint8(m);                \
  TF_CALL_int64(m);                \
  TF_CALL_float(m);                \
  TF_CALL_double(m);               \
  TF_CALL_complex64(m);            \
  TF_CALL_complex128(m);           \
  TF_CALL_bool(m);

namespace functor {

template <typename Device, typename T, typename TIndex>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static absl::Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::UnalignedScalar num_true);
};

template <typename Device, int NDIM, typename T, typename TIndex>
struct Where {
  // Copies indices of true values in input into output.  The pointer
  // found_true should sit on the host.  Compute should copy the
  // number of true elements found into it.  At the end, if
  //   *found_true != output.dimension(0),
  // then the input may have changed between the initial counting of
  // the true values and the call to Where.
  EIGEN_ALWAYS_INLINE static absl::Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64_t>::Matrix output, TIndex* found_true);
};

}  // namespace functor

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_WHERE_OP_H_
