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

#include "machina/core/kernels/reduction_ops_common.h"

namespace machina {

#define REGISTER_CPU_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Mean")                                                      \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<int32>("Tidx"),                               \
      ReductionOp<CPUDevice, type, int32, functor::MeanReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Mean")                                                      \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<int64_t>("Tidx"),                             \
      ReductionOp<CPUDevice, type, int64, functor::MeanReducer<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Mean")                                                      \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<int32>("Tidx")                                \
          .HostMemory("reduction_indices"),                             \
      ReductionOp<GPUDevice, type, int32, functor::MeanReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Mean")                                                      \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<int64_t>("Tidx")                              \
          .HostMemory("reduction_indices"),                             \
      ReductionOp<GPUDevice, type, int64, functor::MeanReducer<type>>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#if GOOGLE_CUDA
TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNELS);
#endif
#undef REGISTER_GPU_KERNELS

#endif

}  // namespace machina
