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

#include "machina/core/kernels/linalg/einsum_op_impl.h"

namespace machina {

#define REGISTER_EINSUM(D, TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Einsum").Device(DEVICE_##D).TypeConstraint<TYPE>("T"), \
      EinsumOp<D##Device, TYPE>);

#define REGISTER_CPU(TYPE) REGISTER_EINSUM(CPU, TYPE)
TF_CALL_complex128(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define REGISTER_GPU(TYPE) REGISTER_EINSUM(GPU, TYPE)
// TODO(rocm): Enable once complex types are supported.
#if GOOGLE_CUDA
TF_CALL_complex128(REGISTER_GPU);
#endif
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

#undef REGISTER_EINSUM

}  // namespace machina
