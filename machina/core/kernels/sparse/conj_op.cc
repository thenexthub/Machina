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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/tensor_util.h"
#include "machina/core/framework/variant_op_registry.h"
#include "machina/core/kernels/cwise_ops.h"
#include "machina/core/kernels/cwise_ops_common.h"
#include "machina/core/kernels/sparse/kernels.h"
#include "machina/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/util/cuda_sparse.h"
#include "machina/core/util/gpu_solvers.h"
#endif

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
template <typename Device, typename T>
class CSRSparseMatrixConjFunctor {
 public:
  explicit CSRSparseMatrixConjFunctor(OpKernelContext* ctx) : ctx_(ctx) {}

  absl::Status operator()(const CSRSparseMatrix& a, CSRSparseMatrix* b) {
    const int total_nnz = a.total_nnz();
    Tensor b_values_t;
    TF_RETURN_IF_ERROR(ctx_->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({total_nnz}), &b_values_t));
    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(),
        a.row_pointers(), a.col_indices(), b_values_t, b));

    const Device& d = ctx_->eigen_device<Device>();
    functor::UnaryFunctor<Device, functor::conj<T>> func;
    func(d, b->values().flat<T>() /*out*/, a.values().flat<T>() /*in*/);

    return absl::OkStatus();
  }

 private:
  OpKernelContext* ctx_;
};

// Partial specialization for real types where conjugation is a noop.
#define NOOP_CONJ_FUNCTOR(T)                                             \
  template <typename Device>                                             \
  class CSRSparseMatrixConjFunctor<Device, T> {                          \
   public:                                                               \
    explicit CSRSparseMatrixConjFunctor(OpKernelContext* ctx) {}         \
    Status operator()(const CSRSparseMatrix& a, CSRSparseMatrix* b) {    \
      TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(         \
          DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(), \
          a.row_pointers(), a.col_indices(), a.values(), b));            \
      return OkStatus();                                                 \
    }                                                                    \
  };

NOOP_CONJ_FUNCTOR(float);
NOOP_CONJ_FUNCTOR(double);

#undef NOOP_CONJ_FUNCTOR

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    CONJ_VARIANT_UNARY_OP, DEVICE_CPU, CSRSparseMatrix,
    (CSRSparseMatrixUnaryHelper<CPUDevice, CSRSparseMatrixConjFunctor>));

#if GOOGLE_CUDA || MACHINA_USE_ROCM

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    CONJ_VARIANT_UNARY_OP, DEVICE_GPU, CSRSparseMatrix,
    (CSRSparseMatrixUnaryHelper<GPUDevice, CSRSparseMatrixConjFunctor>));

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
