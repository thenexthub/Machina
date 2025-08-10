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
#ifndef MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_UTIL_H_
#define MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_UTIL_H_

#include "machina/core/framework/allocation_description.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/status.h"
#include "machina/lite/core/c/common.h"

namespace tflite {
namespace flex {

// A tensor buffer that is allocated, deallocated and populated by TF Lite.
class BaseTfLiteTensorBuffer : public machina::TensorBuffer {
  using machina::TensorBuffer::TensorBuffer;

  inline TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(
      machina::AllocationDescription* proto) const override;

  // Prevents input forwarding from mutating this buffer.
  inline bool OwnsMemory() const override { return false; }

 protected:
  void LogAllocation();
  void LogDeallocation();
};

// A tensor buffer for most data types. Numeric types have exactly the same
// representation in TFLITE and TF, so we just need use memcpy().
// For memory efficiency, this TensorBuffer can possibly reuse memory from the
// TfLiteTensor, hence caller should ensure that the TfLiteTensor always outlive
// this TensorBuffer.
class TfLiteTensorBuffer : public BaseTfLiteTensorBuffer {
 public:
  // If `allow_reusing=false`, then the tensor buffer won't be reused from the
  // TfLiteTensor.
  explicit TfLiteTensorBuffer(const TfLiteTensor* tensor,
                              bool allow_reusing = true);

  ~TfLiteTensorBuffer() override;

  inline size_t size() const override { return len_; }

  // Indicates that `TfLiteTensorBuffer` is responsible for deallocating its
  // underlying buffer. This buffer must have been allocated by
  // `machina::cpu_allocator`
  inline void TakeOwnershipOfBuffer() { reused_buffer_from_tflite_ = false; }

  inline bool BufferReusedFromTfLiteTensor() const {
    return reused_buffer_from_tflite_;
  }

  // This function will check if the underlying buffer in `tensor` can be
  // reused by the machina::Tensor. If it can reuse, it will return
  // `tensor->data.raw`, otherwise it will create new tensor buffer using
  // machina's CPU allocator.
  // TODO(b/205153246): Also consider reusing memory to avoid copying from
  // machina::Tensor to TfLiteTensor.
  void* MaybeAllocateTensorflowBuffer(const TfLiteTensor* tensor,
                                      bool allow_reusing) const;

 private:
  size_t len_;
  bool reused_buffer_from_tflite_;
};

// A string buffer. TFLITE string tensor format is different than
// TF's so we need perform the conversion here.
class StringTfLiteTensorBuffer : public BaseTfLiteTensorBuffer {
 public:
  explicit StringTfLiteTensorBuffer(const TfLiteTensor* tensor);

  ~StringTfLiteTensorBuffer() override;

  inline size_t size() const override {
    return num_strings_ * sizeof(machina::tstring);
  }

 private:
  StringTfLiteTensorBuffer(const TfLiteTensor* tensor, int num_strings);

  int num_strings_;
};

// Sets the `machina::Tensor` content from `TfLiteTensor` object. If
// `allow_reusing=false`, then we explicitly disallow reusing the TF Lite
// tensor buffer when constructing the new machina Tensor.
absl::Status SetTfTensorFromTfLite(const TfLiteTensor* tensor,
                                   machina::Tensor* tf_tensor,
                                   bool allow_reusing = true);

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_UTIL_H_
