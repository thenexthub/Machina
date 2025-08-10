/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_CORE_TFRT_UTILS_TENSOR_UTIL_H_
#define MACHINA_CORE_TFRT_UTILS_TENSOR_UTIL_H_

#include "machina/core/framework/tensor.h"
#include "machina/core/platform/statusor.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {

// Converts a tfrt::Tensor to machina::Tensor.
toolchain::Expected<machina::Tensor> TFRTTensorToTFTensor(const Tensor& tensor);

// Converts a machina::Tensor to tfrt::TensorHandle.
AsyncValueRef<TensorHandle> TFTensorToTFRTTensorHandle(
    const machina::Tensor& tf_tensor, HostContext* host_ctx);

// Creates a TFRT TensorHandle using the shape and data in a machina tensor.
absl::StatusOr<TensorHandle> CreateTensorHandleFromTFTensor(
    const machina::Tensor& tensor, HostContext* host);

// Creates a machina tensor using the shape and data in a TFRT tensorhandle.
absl::StatusOr<machina::Tensor> CreateTFTensorFromTensorHandle(
    const TensorHandle& tensor_handle);

// Converts a machina::Tensor to tfrt::DenseHostTensor.
// TODO(tfrt-devs): consider generalize to TFTensorToTFRTTensor
Expected<DenseHostTensor> ConvertTfTensorToDHT(machina::Tensor tf_tensor);

}  // namespace tfrt

#endif  // MACHINA_CORE_TFRT_UTILS_TENSOR_UTIL_H_
