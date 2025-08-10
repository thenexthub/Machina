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

#ifndef MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_
#define MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_

#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/runtime_fallback/util/type_util.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace machina::tfd {

// Retrieves TFRT TensorMetadata from a machina::Tensor.
inline tfrt::TensorMetadata GetTensorMetadata(
    const machina::Tensor& tf_tensor) {
  auto dtype = tfd::GetTfrtDtype(tf_tensor.dtype());
  auto dim_sizes = tf_tensor.shape().dim_sizes();
  static_assert(sizeof(tfrt::Index) == sizeof(dim_sizes.front()),
                "Invalid dimension type size");
  auto shape = toolchain::ArrayRef(reinterpret_cast<tfrt::Index*>(dim_sizes.data()),
                              dim_sizes.size());
  return tfrt::TensorMetadata(dtype, shape);
}

}  // namespace machina::tfd

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_
