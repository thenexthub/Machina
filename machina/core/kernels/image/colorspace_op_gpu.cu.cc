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
#include "machina/core/kernels/image/colorspace_op.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

#define INSTANTIATE_GPU(T)                        \
  template class functor::RGBToHSV<GPUDevice, T>; \
  template class functor::HSVToRGB<GPUDevice, T>;
TF_CALL_float(INSTANTIATE_GPU);
TF_CALL_double(INSTANTIATE_GPU);
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
