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

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_

#include <cstdint>
#include <memory>

#include "machina/c/tf_tensor.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/runtime_fallback/util/tensor_metadata.h"  // IWYU pragma: export
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace machina {
namespace tfd {

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};
using OwnedTFTensor = std::unique_ptr<TF_Tensor, TFTensorDeleter>;

// Moves one ref on HostBuffer to machina::Tensor.
machina::Tensor MoveHostBufferToTfTensor(
    tfrt::RCReference<tfrt::HostBuffer> host_buffer, tfrt::DType dtype,
    const tfrt::TensorShape& shape);

// Creates a machina::Tensor based on StringHostTensor.
machina::Tensor CopyShtToTfTensor(const tfrt::StringHostTensor& sht);

// Converts tfrt shape to machina shape.
inline machina::TensorShape GetTfShape(const tfrt::TensorShape& shape) {
  toolchain::SmallVector<tfrt::Index, 4> dimensions;
  shape.GetDimensions(&dimensions);
  toolchain::SmallVector<int64_t, 4> dims(dimensions.begin(), dimensions.end());
  return machina::TensorShape(dims);
}

inline void CheckBoolCompatibility() {
  // sizeof(bool) is implementation defined. The following may only work when
  // sizeof(bool) is 1.
  //
  // TODO(tfrt-devs): It is still undefined behavior to directly cast char*
  // between bool* and access the data. Consider allocating target objects and
  // using memcpy instead.
  static_assert(sizeof(bool) == 1, "Only support when bool is 1 byte.");
}

}  // namespace tfd
}  // namespace machina
#endif  // MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
