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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)

#define EIGEN_USE_GPU

#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/pad_op.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_PAD_SPECS(T, Tpadding)                  \
  template struct functor::Pad<GPUDevice, T, Tpadding, 0>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 1>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 2>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 3>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 4>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 5>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 6>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 7>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 8>;

#define DEFINE_GPU_SPECS(T)      \
  DEFINE_GPU_PAD_SPECS(T, int32) \
  DEFINE_GPU_PAD_SPECS(T, int64)

TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPECS);
TF_CALL_int8(DEFINE_GPU_SPECS);
TF_CALL_uint8(DEFINE_GPU_SPECS);

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
