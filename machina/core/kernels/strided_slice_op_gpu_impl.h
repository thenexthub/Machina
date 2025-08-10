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

#ifndef MACHINA_CORE_KERNELS_STRIDED_SLICE_OP_GPU_IMPL_H_
#define MACHINA_CORE_KERNELS_STRIDED_SLICE_OP_GPU_IMPL_H_

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define EIGEN_USE_GPU

#include "machina/core/kernels/strided_slice_op.h"

#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/platform/types.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                                   \
  template struct functor::StridedSlice<GPUDevice, T, 1>;       \
  template struct functor::StridedSlice<GPUDevice, T, 2>;       \
  template struct functor::StridedSlice<GPUDevice, T, 3>;       \
  template struct functor::StridedSlice<GPUDevice, T, 4>;       \
  template struct functor::StridedSlice<GPUDevice, T, 5>;       \
  template struct functor::StridedSlice<GPUDevice, T, 6>;       \
  template struct functor::StridedSlice<GPUDevice, T, 7>;       \
  template struct functor::StridedSlice<GPUDevice, T, 8>;       \
  template struct functor::StridedSliceGrad<GPUDevice, T, 1>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 2>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 3>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 4>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 5>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 6>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 7>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 8>;   \
  template struct functor::StridedSliceAssign<GPUDevice, T, 1>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 2>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 3>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 4>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 5>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 6>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 7>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 8>; \
  template struct functor::StridedSliceAssignScalar<GPUDevice, T>;

}  // end namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
#endif  // MACHINA_CORE_KERNELS_STRIDED_SLICE_OP_GPU_IMPL_H_
