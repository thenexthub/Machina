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

#include "machina/core/kernels/cast_op_impl.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetCpuCastFromDouble(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, double);
  CAST_CASE(CPUDevice, double, float8_e5m2);
  CAST_CASE(CPUDevice, double, float8_e4m3fn);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
CastFunctorType GetGpuCastFromDouble(DataType dst_dtype) {
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
  CAST_CASE(GPUDevice, double, bfloat16);
#else
  CURRY_TYPES3(CAST_CASE, GPUDevice, double);
#endif
  CAST_CASE(GPUDevice, double, float8_e5m2);
  CAST_CASE(GPUDevice, double, float8_e4m3fn);
  return nullptr;
}
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM


}  // namespace machina
