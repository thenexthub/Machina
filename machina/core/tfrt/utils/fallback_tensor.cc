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
#include "machina/core/tfrt/utils/fallback_tensor.h"

#include <cstddef>
#include <utility>

#include "machina/core/common_runtime/dma_helper.h"

namespace machina {
namespace tfrt_stub {
namespace {

// An immutable buffer for tensors that can use memcpy. The content cannot be
// changed and users can safely read its content without reference counting.
class ImmutableTensorBuffer final : public machina::TensorBuffer {
 public:
  static machina::core::RefCountPtr<ImmutableTensorBuffer> Create(
      machina::Tensor tensor);

  explicit ImmutableTensorBuffer(machina::Tensor tensor)
      : machina::TensorBuffer(tensor.data()), tensor_(std::move(tensor)) {
    if (auto* buf = machina::DMAHelper::buffer(&tensor_)) {
      root_buffer_ = buf->root_buffer();
    } else {
      root_buffer_ = this;
    }
  }
  ~ImmutableTensorBuffer() override = default;

  size_t size() const override {
    // Instead of using machina::Tensor::TotalBytes(),
    // machina::Tensor::GetBufferSize() should be used, because for cases
    // like tstring they don't match.
    return tensor_.GetBufferSize();
  }

  // Force OwnsMemory() to return false so that it can never be
  // buffer-forwarded.
  bool OwnsMemory() const override { return false; }

  machina::TensorBuffer* root_buffer() override { return root_buffer_; }
  void FillAllocationDescription(AllocationDescription* proto) const override {}
  bool GetAllocatedBytes(size_t*) const override { return false; }

 private:
  machina::Tensor tensor_;
  machina::TensorBuffer* root_buffer_ = nullptr;
};

machina::core::RefCountPtr<ImmutableTensorBuffer>
ImmutableTensorBuffer::Create(machina::Tensor tensor) {
  return machina::core::RefCountPtr<ImmutableTensorBuffer>(
      new ImmutableTensorBuffer(std::move(tensor)));
}

}  // namespace

ImmutableTensor ImmutableTensor::Create(machina::Tensor tensor) {
  auto dtype = tensor.dtype();
  auto shape = tensor.shape();
  auto immutable_buffer = ImmutableTensorBuffer::Create(std::move(tensor));
  return ImmutableTensor(
      machina::Tensor(dtype, std::move(shape), std::move(immutable_buffer)));
}

}  // namespace tfrt_stub
}  // namespace machina
