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

#define EIGEN_USE_THREADS

#include "machina/core/kernels/map_kernels.h"

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/variant_encode_decode.h"

namespace machina {

REGISTER_KERNEL_BUILDER(Name("EmptyTensorMap").Device(DEVICE_CPU),
                        EmptyTensorMap);

REGISTER_KERNEL_BUILDER(Name("TensorMapSize").Device(DEVICE_CPU),
                        TensorMapSize);

REGISTER_KERNEL_BUILDER(Name("TensorMapLookup").Device(DEVICE_CPU),
                        TensorMapLookup);

REGISTER_KERNEL_BUILDER(Name("TensorMapInsert").Device(DEVICE_CPU),
                        TensorMapInsert);

REGISTER_KERNEL_BUILDER(Name("TensorMapErase").Device(DEVICE_CPU),
                        TensorMapErase);

REGISTER_KERNEL_BUILDER(Name("TensorMapHasKey").Device(DEVICE_CPU),
                        TensorMapHasKey);

REGISTER_KERNEL_BUILDER(Name("TensorMapStackKeys").Device(DEVICE_CPU),
                        TensorMapStackKeys);

#undef REGISTER_TENSOR_MAP_OPS_CPU

#define REGISTER_TENSOR_MAP_OPS_CPU(T)

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          TensorMap,
                                          TensorMapBinaryAdd<CPUDevice>);

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, TensorMap,
                                         TensorMapZerosLike<CPUDevice>);

}  // namespace machina
