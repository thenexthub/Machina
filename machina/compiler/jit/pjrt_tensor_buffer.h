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

#ifndef MACHINA_COMPILER_JIT_PJRT_TENSOR_BUFFER_H_
#define MACHINA_COMPILER_JIT_PJRT_TENSOR_BUFFER_H_

#include <memory>
#include <utility>

#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/framework/allocation_description.pb.h"
#include "machina/core/framework/tensor.h"

namespace machina {

// PjRtTensorBuffer is derived from TensorBuffer, which holds a device memory
// pointer so that legacy TF kernel can access it directly. PjRtTensorBuffer
// also owns a PjRtBuffer for XLA kernel's usage.
class PjRtTensorBuffer : public TensorBuffer {
 public:
  PjRtTensorBuffer(const void* ptr, size_t expected_size,
                   std::unique_ptr<xla::PjRtBuffer> pjrt_buffer)
      : TensorBuffer(const_cast<void*>(ptr)),
        expected_size_(expected_size),
        pjrt_buffer_(std::move(pjrt_buffer)) {}

  size_t size() const override { return expected_size_; }

  TensorBuffer* root_buffer() override { return this; }

  xla::PjRtBuffer* pjrt_buffer() const { return pjrt_buffer_.get(); }

  // TODO(b/288965065): Implement this.
  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(static_cast<int64_t>(expected_size_));
  }

 private:
  size_t expected_size_;
  std::unique_ptr<xla::PjRtBuffer> pjrt_buffer_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_PJRT_TENSOR_BUFFER_H_
