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
#ifndef MACHINA_CORE_KERNELS_TILE_OPS_CPU_IMPL_H_
#define MACHINA_CORE_KERNELS_TILE_OPS_CPU_IMPL_H_

#define EIGEN_USE_THREADS

#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/tile_ops_impl.h"

namespace machina {

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Register functors used for TileGradientOp.
#define DEFINE_DIM(T, NDIM)                     \
  template struct TileGrad<CPUDevice, T, NDIM>; \
  template struct ReduceAndReshape<CPUDevice, T, NDIM, 1>;
#define DEFINE_TYPE(T) DEFINE_DIM(T, CPU_PROVIDED_IXDIM)

TF_CALL_float(DEFINE_TYPE);
TF_CALL_bfloat16(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);
TF_CALL_half(DEFINE_TYPE);
TF_CALL_complex64(DEFINE_TYPE);
TF_CALL_complex128(DEFINE_TYPE);

#undef DEFINE_DIM
#undef DEFINE_TYPE


}  // end namespace functor
}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_TILE_OPS_CPU_IMPL_H_
