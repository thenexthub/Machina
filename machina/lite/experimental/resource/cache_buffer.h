/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_LITE_EXPERIMENTAL_RESOURCE_CACHE_BUFFER_H_
#define MACHINA_LITE_EXPERIMENTAL_RESOURCE_CACHE_BUFFER_H_

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "machina/lite/core/c/common.h"
#include "machina/lite/experimental/resource/resource_variable.h"
#include "machina/lite/kernels/kernel_util.h"

namespace tflite {
namespace resource {

/// WARNING: Experimental interface, subject to change.
// A Cache Buffer class. Useful for keeping the keys and values of a
// transformer block attention mechanism in autoregressive decode.
// Ops can access this buffer and add tensors to it. It also keeps track of the
// number of used entries in the cache.
class CacheBuffer : public ResourceVariable {
 public:
  CacheBuffer() = default;
  CacheBuffer(const CacheBuffer &) = delete;
  ~CacheBuffer() override;
  CacheBuffer &operator=(const CacheBuffer &) = delete;
  // Initialize tensor of a certain shape using the provided type.
  TfLiteStatus Initialize(const TfLiteIntArray &shape);
  size_t GetNumEntries(int idx) const;
  float *GetBuffer();
  size_t GetSize();
  void SetNumEntries(int idx, size_t count);

 private:
  // The number of entries currently used in the buffer;
  std::unique_ptr<size_t[]> num_entries_;
  // The float buffer for storage. Has shape:
  // <batch, num layers, seq length, num heads, head dim>
  std::unique_ptr<float[]> buffer_;
  TfLiteIntArray *dims_;
};

}  // namespace resource
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_RESOURCE_CACHE_BUFFER_H_
