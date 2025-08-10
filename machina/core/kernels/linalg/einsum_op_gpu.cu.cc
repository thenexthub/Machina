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

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define EIGEN_USE_GPU

#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/linalg/einsum_op.h"

namespace machina {

#define DECLARE_GPU_SPECS_NDIM(T, NDIM)                              \
  template struct functor::StrideFunctor<Eigen::GpuDevice, T, NDIM>; \
  template struct functor::InflateFunctor<Eigen::GpuDevice, T, NDIM>;

#define DECLARE_GPU_SPECS(T)    \
  DECLARE_GPU_SPECS_NDIM(T, 1); \
  DECLARE_GPU_SPECS_NDIM(T, 2); \
  DECLARE_GPU_SPECS_NDIM(T, 3); \
  DECLARE_GPU_SPECS_NDIM(T, 4); \
  DECLARE_GPU_SPECS_NDIM(T, 5); \
  DECLARE_GPU_SPECS_NDIM(T, 6);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS_NDIM
#undef DECLARE_GPU_SPECS

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
