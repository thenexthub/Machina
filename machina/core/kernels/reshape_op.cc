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

// See docs in ../ops/array_ops.cc.
#include "machina/core/kernels/reshape_op.h"

namespace machina {

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_CPU)
                            .HostMemory("shape")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_CPU)
                            .HostMemory("shape")
                            .TypeConstraint<int64_t>("Tshape"),
                        ReshapeOp);

#define REGISTER_GPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                         \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("shape")                \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("Tshape"),   \
                          ReshapeOp);                             \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                         \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("shape")                \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int64_t>("Tshape"), \
                          ReshapeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tshape"),
                        ReshapeOp);
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                         \
                              .Device(DEVICE_DEFAULT)             \
                              .HostMemory("shape")                \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("Tshape"),   \
                          ReshapeOp);                             \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                         \
                              .Device(DEVICE_DEFAULT)             \
                              .HostMemory("shape")                \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int64_t>("Tshape"), \
                          ReshapeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tshape"),
                        ReshapeOp);

}  // namespace machina
