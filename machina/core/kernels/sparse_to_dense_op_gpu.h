/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#if !GOOGLE_CUDA && !MACHINA_USE_ROCM
#error This file must only be included when building with Cuda
#endif

#ifndef MACHINA_CORE_KERNELS_SPARSE_TO_DENSE_OP_GPU_H_
#define MACHINA_CORE_KERNELS_SPARSE_TO_DENSE_OP_GPU_H_

#include "machina/xla/stream_executor/device_memory.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

namespace functor {
template <typename T, typename Index>
struct LaunchSparseToDense {
  void operator()(OpKernelContext* c, AsyncOpKernel::DoneCallback done,
                  AsyncOpKernel* op, bool validate_indices,
                  const Tensor& indices, const Tensor& values,
                  const Tensor& shape, const T default_value, Tensor* dense);
};

}  // namespace functor

}  // namespace machina
#endif  // MACHINA_CORE_KERNELS_SPARSE_TO_DENSE_OP_GPU_H_
