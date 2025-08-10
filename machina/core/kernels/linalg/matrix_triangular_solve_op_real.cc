/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/core/kernels/linalg/matrix_triangular_solve_op_impl.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace machina {

TF_CALL_float(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_CPU);
TF_CALL_double(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_CPU);

#if GOOGLE_CUDA || MACHINA_USE_ROCM
TF_CALL_float(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
TF_CALL_double(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
