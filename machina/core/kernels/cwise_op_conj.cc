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

#include "machina/core/framework/variant_op_registry.h"
#include "machina/core/kernels/cwise_ops_common.h"

namespace machina {

REGISTER2(UnaryOp, CPU, "Conj", functor::conj, complex64, complex128);

REGISTER_VARIANT(UnaryVariantOp, CPU, "Conj", CONJ_VARIANT_UNARY_OP);

#if GOOGLE_CUDA || MACHINA_USE_ROCM
REGISTER_KERNEL_BUILDER(
    Name("Conj").Device(DEVICE_GPU).TypeConstraint<Variant>("T"),
    UnaryVariantOp<GPUDevice, CONJ_VARIANT_UNARY_OP>);
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER_KERNEL_BUILDER(
    Name("Conj").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    UnaryOp<GPUDevice, functor::conj<complex64>>);
REGISTER_KERNEL_BUILDER(
    Name("Conj").Device(DEVICE_GPU).TypeConstraint<complex128>("T"),
    UnaryOp<GPUDevice, functor::conj<complex128>>);
#endif
#endif
}  // namespace machina
