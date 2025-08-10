/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_NEG_H_
#define MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_NEG_H_

#include "machina/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void Negate(const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = -input_data[i];
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_NEG_H_
