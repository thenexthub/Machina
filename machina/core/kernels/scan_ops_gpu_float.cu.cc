/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/core/kernels/scan_ops.h"
#include "machina/core/kernels/scan_ops_gpu.h"

namespace machina {
using Eigen::GpuDevice;
template struct functor::Scan<GpuDevice, Eigen::internal::SumReducer<float>,
                              float>;
template struct functor::Scan<GpuDevice, Eigen::internal::ProdReducer<float>,
                              float>;
template struct functor::Scan<GpuDevice, functor::LogSumExpReducer<float>,
                              float>;
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
