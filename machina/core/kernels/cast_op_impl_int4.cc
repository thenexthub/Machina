/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include <cstdint>

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/kernels/cast_op.h"
#include "machina/core/kernels/cast_op_impl.h"
#include "machina/core/platform/types.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetCpuCastFromInt4(DataType dst_dtype) {
  // Only allow casts to integer types.
  CAST_CASE(CPUDevice, int4, int4);
  CAST_CASE(CPUDevice, int4, int8);
  CAST_CASE(CPUDevice, int4, int16);
  CAST_CASE(CPUDevice, int4, int32);
  CAST_CASE(CPUDevice, int4, int64_t);
  CAST_CASE(CPUDevice, int4, uint4);
  CAST_CASE(CPUDevice, int4, uint8);
  CAST_CASE(CPUDevice, int4, uint16);
  CAST_CASE(CPUDevice, int4, uint32);
  CAST_CASE(CPUDevice, int4, uint64_t);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
CastFunctorType GetGpuCastFromInt4(DataType dst_dtype) {
  // Only allow casts to integer types.
  CAST_CASE(GPUDevice, int4, int4);
  CAST_CASE(GPUDevice, int4, int8);
  CAST_CASE(GPUDevice, int4, int16);
  CAST_CASE(GPUDevice, int4, int32);
  CAST_CASE(GPUDevice, int4, int64_t);
  CAST_CASE(GPUDevice, int4, uint4);
  CAST_CASE(GPUDevice, int4, uint8);
  CAST_CASE(GPUDevice, int4, uint16);
  CAST_CASE(GPUDevice, int4, uint32);
  CAST_CASE(GPUDevice, int4, uint64_t);
  return nullptr;
}
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
