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

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#include "machina/core/framework/register_types.h"
#include "machina/core/kernels/segment_reduction_ops_gpu.cu.h"

namespace machina {
namespace functor {

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Lowest<T>, functor::Max>;    \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Highest<T>, functor::Min>;   \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::Prod>;

// Sum is the only op that supports all input types currently.
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index)         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, \
                                         functor::Zero<T>, functor::Sum>;

#define DEFINE_REAL_GPU_SPECS(T)                  \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int64_t);

#define DEFINE_SUM_GPU_SPECS(T)                  \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int64_t);

TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);

// TODO(rocm): support atomicAdd for complex numbers on ROCm
#if GOOGLE_CUDA
TF_CALL_COMPLEX_TYPES(DEFINE_SUM_GPU_SPECS);
#endif

#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

}  // namespace functor
}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
