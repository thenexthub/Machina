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

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define EIGEN_USE_GPU

#include "machina/core/kernels/scatter_functor_gpu.cu.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

// Instantiates functor specializations for GPU.
#define DEFINE_GPU_SPECS_OP(T, Index, op)                           \
  template struct functor::ScatterFunctor<GPUDevice, T, Index, op>; \
  template struct functor::ScatterScalarFunctor<GPUDevice, T, Index, op>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)                       \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN); \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MUL);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::DIV);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MIN);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MAX);

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

DEFINE_GPU_SPECS(Eigen::half);
DEFINE_GPU_SPECS(Eigen::bfloat16);
DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
// TODO: The following fails to compile.
// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX
#undef DEFINE_GPU_SPECS_OP

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
