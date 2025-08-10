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

#include "machina/core/framework/variant_op_registry.h"
#include "machina/core/kernels/sparse/sparse_matrix.h"

namespace machina {

constexpr const char CSRSparseMatrix::kTypeName[];

// Register variant decoding function for TF's RPC.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CSRSparseMatrix,
                                       CSRSparseMatrix::kTypeName);

#define REGISTER_CSR_COPY(DIRECTION)                    \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      CSRSparseMatrix, DIRECTION, CSRSparseMatrix::DeviceCopy)

REGISTER_CSR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

#undef REGISTER_CSR_COPY

}  // namespace machina
