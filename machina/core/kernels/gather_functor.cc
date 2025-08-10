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

#include "machina/core/kernels/gather_functor.h"

#include "machina/core/framework/register_types.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPECS_INDEX(T, Index)                               \
  template <>                                                           \
  int64_t GatherFunctor<GPUDevice, T, Index>::operator()(               \
      OpKernelContext* ctx, typename TTypes<T, 3>::ConstTensor Tparams, \
      typename TTypes<Index>::ConstFlat Tindices,                       \
      typename TTypes<T, 3>::Tensor Tout);                              \
  extern template struct GatherFunctor<GPUDevice, T, Index>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64_t)

TF_CALL_int64(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace machina

#else

#include "machina/core/kernels/gather_functor.h"

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
