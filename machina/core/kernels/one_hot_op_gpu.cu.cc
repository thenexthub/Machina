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

// See docs in ../ops/array_ops.cc

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)

#define EIGEN_USE_GPU

#include "machina/core/framework/register_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/one_hot_op.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC_INDEX(T, TI)             \
  template class generator::OneGenerator<T, TI>; \
  template struct functor::OneHot<GPUDevice, T, TI>;

#define DEFINE_GPU_SPEC(T)         \
  DEFINE_GPU_SPEC_INDEX(T, uint8); \
  DEFINE_GPU_SPEC_INDEX(T, int8);  \
  DEFINE_GPU_SPEC_INDEX(T, int32); \
  DEFINE_GPU_SPEC_INDEX(T, int64)

TF_CALL_int8(DEFINE_GPU_SPEC);
TF_CALL_int32(DEFINE_GPU_SPEC);
TF_CALL_int64(DEFINE_GPU_SPEC);
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPEC);

#undef DEFINE_GPU_SPEC_INDEX
#undef DEFINE_GPU_SPEC

}  // end namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
