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

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "machina/core/kernels/conv_2d.h"
#include "machina/core/kernels/conv_2d_gpu.h"

namespace machina {

namespace functor {

template struct SwapDimension1And2InTensor3<Eigen::GpuDevice, Eigen::half>;

// For 2d ops.
template struct TransformFilter<Eigen::GpuDevice, Eigen::half, int, 4>;
template struct ReverseTransformFilter<Eigen::GpuDevice, Eigen::half, 4>;
template struct NHWCToNCHW<Eigen::GpuDevice, Eigen::half, 4>;
template struct NCHWToNHWC<Eigen::GpuDevice, Eigen::half, 4>;
template struct PadInput<Eigen::GpuDevice, Eigen::half, int, 4>;

// For 3d ops.
template struct TransformFilter<Eigen::GpuDevice, Eigen::half, int, 5>;
template struct ReverseTransformFilter<Eigen::GpuDevice, Eigen::half, 5>;
template struct NHWCToNCHW<Eigen::GpuDevice, Eigen::half, 5>;
template struct NCHWToNHWC<Eigen::GpuDevice, Eigen::half, 5>;
template struct PadInput<Eigen::GpuDevice, Eigen::half, int, 5>;

}  // namespace functor
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
