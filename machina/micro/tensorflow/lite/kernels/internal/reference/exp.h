/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_EXP_H_
#define MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_EXP_H_

#include <cmath>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "machina/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void Exp(const T* input_data, const size_t num_elements,
                T* output_data) {
  ruy::profiler::ScopeLabel label("Exp");
  for (size_t idx = 0; idx < num_elements; ++idx) {
    output_data[idx] = std::exp(input_data[idx]);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_EXP_H_
