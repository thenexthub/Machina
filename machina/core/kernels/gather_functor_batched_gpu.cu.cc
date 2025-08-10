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

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define EIGEN_USE_GPU

#include "machina/core/kernels/gather_functor_batched_gpu.cu.h"

#include "machina/core/framework/register_types.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPECS_INDEX(T, Index) \
  template struct functor::GatherFunctorBatched<GPUDevice, T, Index>

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_int32(DEFINE_GPU_SPECS);
TF_CALL_int64(DEFINE_GPU_SPECS);
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPECS);
TF_CALL_float8_e5m2(DEFINE_GPU_SPECS);
TF_CALL_float8_e4m3fn(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
