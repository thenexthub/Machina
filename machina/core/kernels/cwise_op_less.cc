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

#include "machina/core/kernels/cwise_ops_common.h"

namespace machina {
REGISTER5(BinaryOp, CPU, "Less", functor::less, float, Eigen::half, double,
          bfloat16, int32);
REGISTER7(BinaryOp, CPU, "Less", functor::less, uint8, uint16, uint32, uint64,
          int8, int16, int64_t);

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER9(BinaryOp, GPU, "Less", functor::less, float, Eigen::half, double,
          int64, uint8, uint16, uint32, uint64, int8);
REGISTER(BinaryOp, GPU, "Less", functor::less, int16);
#endif
REGISTER(BinaryOp, GPU, "Less", functor::less, bfloat16);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Less")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less<int32>>);
#endif
REGISTER_KERNEL_BUILDER(Name("Less")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less<int32>>);
}  // namespace machina
