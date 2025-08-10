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

#ifndef MACHINA_CORE_KERNELS_RNN_BLAS_GEMM_H_
#define MACHINA_CORE_KERNELS_RNN_BLAS_GEMM_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/tensor_types.h"
#include "machina/core/kernels/eigen_activations.h"
#include "machina/core/platform/types.h"

#if defined(MACHINA_USE_CUSTOM_CONTRACTION_KERNEL)
#include "machina/xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

namespace machina {
class OpKernelContext;
namespace functor {

template <typename T>
struct TensorCuBlasGemm {
  void operator()(OpKernelContext* ctx, bool transa, bool transb, uint64 m,
                  uint64 n, uint64 k, float alpha, const T* a, int lda,
                  const T* b, int ldb, float beta, T* c, int ldc);
};

template <typename T>
struct gemm_compute_type {
  typedef T type;
};

template <>
struct gemm_compute_type<Eigen::half> {
  typedef float type;
};

template <typename Device, typename T, bool USE_CUBLAS>
struct TensorBlasGemm;

template <typename Device, typename T>
struct TensorBlasGemm<Device, T, true /* USE_CUBLAS */> {
  static void compute(OpKernelContext* ctx, const Device& d, bool transa,
                      bool transb, typename gemm_compute_type<T>::type alpha,
                      typename TTypes<T>::ConstMatrix a,
                      typename TTypes<T>::ConstMatrix b,
                      typename gemm_compute_type<T>::type beta,
                      typename TTypes<T>::Matrix c) {
    int64_t m = c.dimensions()[0];
    int64_t n = c.dimensions()[1];
    int64_t k = transa ? a.dimensions()[0] : a.dimensions()[1];

    TensorCuBlasGemm<T>()(ctx, transb, transa, n, m, k, alpha, b.data(),
                          transb ? k : n, a.data(), transa ? m : k, beta,
                          c.data(), n);
  }
};

template <typename Device, typename T>
struct TensorBlasGemm<Device, T, false /* USE_CUBLAS */> {
  static void compute(OpKernelContext* ctx, const Device& d, bool transa,
                      bool transb, typename gemm_compute_type<T>::type alpha,
                      typename TTypes<T>::ConstMatrix a,
                      typename TTypes<T>::ConstMatrix b,
                      typename gemm_compute_type<T>::type beta,
                      typename TTypes<T>::Matrix c) {
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] =
        Eigen::IndexPair<Eigen::DenseIndex>(transa == false, transb == true);
    if (alpha == typename gemm_compute_type<T>::type(1.f) &&
        beta == typename gemm_compute_type<T>::type(0.f)) {
      c.device(d) = a.contract(b, contract_pairs);
    } else if (alpha == typename gemm_compute_type<T>::type(1.f) &&
               beta == typename gemm_compute_type<T>::type(1.f)) {
      c.device(d) += a.contract(b, contract_pairs);
    } else {
      c.device(d) = c.constant(T(alpha)) * a.contract(b, contract_pairs) +
                    c.constant(T(beta)) * c;
    }
  }
};

}  // namespace functor
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_RNN_BLAS_GEMM_H_
