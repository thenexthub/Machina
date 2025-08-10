/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/jit/pjrt_tensor_buffer_util.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/compiler/jit/pjrt_tensor_buffer.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/common_runtime/dma_helper.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "tsl/platform/statusor.h"

namespace machina {

static size_t GetTensorSize(const TensorShape& shape, const DataType dtype) {
  return shape.num_elements() * DataTypeSize(dtype);
}

absl::StatusOr<Tensor> MakeTensorFromPjRtBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer::ExternalReference> ref,
                      pjrt_buffer->AcquireExternalReference());
  auto* tensor_buffer =
      new PjRtTensorBuffer(ref->OpaqueDeviceMemoryDataPointer(),
                           GetTensorSize(shape, dtype), std::move(pjrt_buffer));
  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  return result;
}

// If existing_tensor does not use PjRtTensorBuffer and the opaque device memory
// is the same, the tensor should be reused so that the same device memory will
// not be double-freed.
static bool ShouldReuseTensor(void* opaque_device_memory,
                              const size_t expected_size,
                              const Tensor* existing_tensor) {
  const PjRtTensorBuffer* input_pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(existing_tensor));
  if (input_pjrt_tensor_buffer != nullptr) {
    return false;
  }

  const size_t current_size =
      GetTensorSize(existing_tensor->shape(), existing_tensor->dtype());
  return existing_tensor->tensor_data().data() == opaque_device_memory &&
         current_size == expected_size;
}

absl::Status PjRtTensorBufferUtil::UpdateOrMakeTensorWithPjRtBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer, Tensor* output_tensor) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer::ExternalReference> ref,
                      pjrt_buffer->AcquireExternalReference());
  const size_t expected_size = GetTensorSize(shape, dtype);
  void* opaque_device_memory = ref->OpaqueDeviceMemoryDataPointer();
  auto* tensor_buffer = new PjRtTensorBuffer(
      opaque_device_memory, expected_size, std::move(pjrt_buffer));
  if (ShouldReuseTensor(opaque_device_memory, expected_size, output_tensor)) {
    output_tensor->buf_ = tensor_buffer;
    return absl::OkStatus();
  }

  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  *output_tensor = result;
  return absl::OkStatus();
}

}  // namespace machina
