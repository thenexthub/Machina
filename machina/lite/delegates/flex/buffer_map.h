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
#ifndef MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_H_
#define MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_H_

#include <map>

#include "machina/core/framework/tensor.h"
#include "machina/lite/core/c/common.h"

namespace tflite {
namespace flex {

// Maps a TF Lite tensor index into a TensorFlow tensor.
//
// The TF Lite interpreter assigns integer indices to each of its tensors, but
// the Flex delegate deals in terms of TensorFlow tensors. This class maps
// from indices to tensors and allows the creation of new tensors to be
// associated with a given index.
class BufferMap {
 public:
  BufferMap();
  ~BufferMap();

  // Returns true if the given 'tensor_index' has a corresponding
  // machina::Tensor.
  bool HasTensor(int tensor_index) const;

  // Returns the machina::Tensor associated with the given 'tensor_index'.
  // Precondition: HasTensor() is true.
  machina::Tensor GetTensor(int tensor_index) const;

  // Returns the const pointer to machina::Tensor associated with the given
  // 'tensor_index'.
  // Precondition: HasTensor() is true.
  const machina::Tensor* GetTensorPtr(int tensor_index) const;

  // Associates the given machina::Tensor with the given 'tensor_index'.
  // Note that TensorFlow Tensors share data buffers, so this method is only a
  // shallow copy.
  void SetFromTensorFlow(int tensor_index, machina::Tensor tensor);

  // Same as above but creates a new machina::Tensor with a copy of the
  // given TfLiteTensor's data. If `allow_reusing=false`, then we explicitly
  // disallow reusing the TF Lite tensor buffer when constructing the new
  // machina Tensor.
  void SetFromTfLite(int tensor_index, const TfLiteTensor* tensor,
                     bool allow_reusing = true);

 private:
  // Mapping from TL Lite tensor ID to TensorFlow's Tensor. All tensors that
  // are inputs or outputs of a subgraph will be added here, irrespective of
  // whether their data are managed by TF Lite or TensorFlow.
  std::map<int, machina::Tensor> id_to_tensor_;
};

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_FLEX_BUFFER_MAP_H_
