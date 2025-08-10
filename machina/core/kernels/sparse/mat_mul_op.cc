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

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/SparseCore"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/type_traits.h"
#include "machina/core/framework/variant_op_registry.h"
#include "machina/core/kernels/cwise_ops_common.h"
#include "machina/core/kernels/dense_update_functor.h"
#include "machina/core/kernels/fill_functor.h"
#include "machina/core/kernels/sparse/kernels.h"
#include "machina/core/kernels/sparse/sparse_matrix.h"
#include "machina/core/kernels/sparse/transpose_op.h"
#include "machina/core/kernels/transpose_functor.h"
#include "machina/core/lib/gtl/inlined_vector.h"
#include "machina/core/platform/threadpool.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/util/cuda_sparse.h"
#include "machina/core/util/gpu_solvers.h"
#endif

#include "machina/core/kernels/sparse/mat_mul_op.h"

namespace machina {

#define REGISTER_CPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseMatrixMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CSRMatMulCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define REGISTER_GPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseMatrixMatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CSRMatMulGPUOp<T>);

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
