/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_DMA_HELPER_H_
#define MACHINA_CORE_COMMON_RUNTIME_DMA_HELPER_H_

#include "machina/core/framework/tensor.h"

namespace machina {

// For TensorFlow internal use only.
class DMAHelper {
 public:
  static bool CanUseDMA(const Tensor* t) { return t->CanUseDMA(); }
  static const void* base(const Tensor* t) { return t->base<const void>(); }
  static void* base(Tensor* t) { return t->base<void>(); }
  static TensorBuffer* buffer(Tensor* t) { return t->buf_; }
  static const TensorBuffer* buffer(const Tensor* t) { return t->buf_; }
  static void UnsafeSetShape(Tensor* t, const TensorShape& s) {
    t->set_shape(s);
  }
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_DMA_HELPER_H_
