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

#ifndef MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_X86_H_
#define MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_X86_H_

// If TFLITE_WITH_RUY is set, Ruy is the only GEMM option. In this header
// we select either Ruy or an alternative based on the SIMD extentions
// available on the given x86 platform.
#ifndef TFLITE_WITH_RUY

#include "machina/lite/kernels/cpu_backend_context.h"
#include "machina/lite/kernels/cpu_backend_gemm_eigen.h"
#include "machina/lite/kernels/cpu_backend_gemm_gemmlowp.h"
#include "machina/lite/kernels/cpu_backend_gemm_params.h"
#include "machina/lite/kernels/cpu_backend_gemm_ruy.h"
#include "machina/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplX86 {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    // TODO(b/168923364) Ruy is preferred on x86, but check if the deprecated
    // path is enabled.
    if (context->PreferGemmlowpOnX86()) {
      // Dispatch to gemmlowp.
      detail::GemmImplUsingGemmlowp<
          LhsScalar, RhsScalar, AccumScalar, DstScalar,
          quantization_flavor>::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                                    dst_params, dst_data, params, context);

      return;
    }
    // Run-time dispatch to Ruy for platforms with AVX or above.
    detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             quantization_flavor>::Run(lhs_params, lhs_data,
                                                       rhs_params, rhs_data,
                                                       dst_params, dst_data,
                                                       params, context);
  }
};

// For float, defer to eigen for now.
template <>
struct GemmImplX86<float, float, float, float,
                   QuantizationFlavor::kFloatingPoint> {
  static void Run(const MatrixParams<float>& lhs_params, const float* lhs_data,
                  const MatrixParams<float>& rhs_params, const float* rhs_data,
                  const MatrixParams<float>& dst_params, float* dst_data,
                  const GemmParams<float, float,
                                   QuantizationFlavor::kFloatingPoint>& params,
                  CpuBackendContext* context) {
    GemmImplUsingEigen::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                            dst_params, dst_data, params, context);
  }
};

// gemmlowp requires NEON for certain quantization cases. See note in
// cpu_backend_gemm.h
#if !defined(GEMMLOWP_NEON)
template <typename SrcScalar, QuantizationFlavor quantization_flavor>
struct GemmImplX86<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                   quantization_flavor>
    : detail::GemmImplUsingRuy<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                               quantization_flavor> {};

template <typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplX86<std::int8_t, std::int8_t, std::int32_t, DstScalar,
                   quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               DstScalar, quantization_flavor> {};

template <QuantizationFlavor quantization_flavor>
struct GemmImplX86<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                   quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               std::int8_t, quantization_flavor> {};
#endif  // not GEMMLOWP_NEON
}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // not TFLITE_WITH_RUY

#endif  // MACHINA_LITE_KERNELS_CPU_BACKEND_GEMM_X86_H_
