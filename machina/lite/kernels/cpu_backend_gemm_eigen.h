/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_EIGEN_H_
#define MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_EIGEN_H_

#ifndef TFLITE_WITH_RUY

#include "machina/lite/kernels/cpu_backend_context.h"
#include "machina/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

struct GemmImplUsingEigen {
  static void Run(const MatrixParams<float>& lhs_params, const float* lhs_data,
                  const MatrixParams<float>& rhs_params, const float* rhs_data,
                  const MatrixParams<float>& dst_params, float* dst_data,
                  const GemmParams<float, float>& params,
                  CpuBackendContext* /* context */);
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // not TFLITE_WITH_RUY

#endif  // MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_EIGEN_H_
