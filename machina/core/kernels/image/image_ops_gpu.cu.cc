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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "machina/core/framework/register_types.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/image/image_ops.h"
#include "machina/core/platform/types.h"

namespace machina {

namespace functor {

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template class FillProjectiveTransform<GPUDevice, uint8>;
template class FillProjectiveTransform<GPUDevice, int32>;
template class FillProjectiveTransform<GPUDevice, int64>;
template class FillProjectiveTransform<GPUDevice, Eigen::half>;
template class FillProjectiveTransform<GPUDevice, float>;
template class FillProjectiveTransform<GPUDevice, double>;

}  // end namespace functor

}  // end namespace machina

#endif  // GOOGLE_CUDA
