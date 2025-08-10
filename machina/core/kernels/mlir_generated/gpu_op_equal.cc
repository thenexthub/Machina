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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/mlir_generated/base_gpu_op.h"

namespace machina {

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_HALF, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_FLOAT, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_DOUBLE, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_BOOL, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_INT8, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_INT16, DT_BOOL);
// TODO(b/25387198): We cannot use a regular GPU kernel for int32.
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_INT64, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(Equal, DT_INT8, DT_BOOL, DT_UINT8,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(Equal, DT_INT16, DT_BOOL, DT_UINT16,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(Equal, DT_INT32, DT_BOOL, DT_UINT32,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(Equal, DT_INT64, DT_BOOL, DT_UINT64,
                                         DT_BOOL);

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_COMPLEX64, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(Equal, DT_COMPLEX128, DT_BOOL);

}  // namespace machina
