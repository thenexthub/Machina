/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include <limits>

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_types.h"
#include "machina/core/framework/variant.h"
#include "machina/core/framework/variant_op_registry.h"
#include "machina/core/kernels/concat_lib.h"
#include "machina/core/kernels/list_kernels.h"
#include "machina/core/lib/core/coding.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/util/util.h"

namespace machina {

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_TENSOR_LIST_OPS_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("TensorListStack")                          \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape"),                \
                          TensorListStack<GPUDevice, T>)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListGather")                         \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("indices")                       \
                              .HostMemory("element_shape"),                \
                          TensorListGather<GPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListGetItem")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("index")                         \
                              .HostMemory("element_shape"),                \
                          TensorListGetItem<GPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListPopBack")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape"),                \
                          TensorListPopBack<GPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListConcat")                         \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("lengths"),                      \
                          TensorListConcat<GPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListConcatV2")                       \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("leading_dims")                  \
                              .HostMemory("element_shape")                 \
                              .HostMemory("lengths"),                      \
                          TensorListConcat<GPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListPushBackBatch")                  \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU),                         \
                          TensorListPushBackBatch<GPUDevice, T>)           \
  REGISTER_KERNEL_BUILDER(Name("TensorListFromTensor")                     \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape"),                \
                          TensorListFromTensor<GPUDevice, T>)              \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatter")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape")                 \
                              .HostMemory("indices"),                      \
                          TensorListScatter<GPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatterV2")                      \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape")                 \
                              .HostMemory("num_elements")                  \
                              .HostMemory("indices"),                      \
                          TensorListScatter<GPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatterIntoExistingList")        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("indices"),                      \
                          TensorListScatterIntoExistingList<GPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("TensorListSplit")                          \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_GPU)                          \
                              .HostMemory("element_shape")                 \
                              .HostMemory("lengths"),                      \
                          TensorListSplit<GPUDevice, T>)

TF_CALL_int32(REGISTER_TENSOR_LIST_OPS_GPU);
TF_CALL_int64(REGISTER_TENSOR_LIST_OPS_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_TENSOR_LIST_OPS_GPU);

#undef REGISTER_TENSOR_LIST_OPS_GPU

REGISTER_KERNEL_BUILDER(Name("TensorListPopBack")
                            .TypeConstraint<Variant>("element_dtype")
                            .Device(DEVICE_GPU)
                            .HostMemory("element_shape"),
                        TensorListPopBack<GPUDevice, Variant>)

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_GPU,
                                          TensorList,
                                          TensorListBinaryAdd<GPUDevice>);
REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_GPU, TensorList,
                                         TensorListZerosLike<GPUDevice>);

}  // namespace machina
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
