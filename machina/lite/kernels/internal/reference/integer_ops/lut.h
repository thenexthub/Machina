/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LUT_H_
#define MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LUT_H_

#include "machina/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

template <typename InputT, typename OutputT>
inline void LookupTable(const InputT* input_data, int num_elements,
                        const OutputT* lut, OutputT* output_data) {
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = LUTLookup(input_data[i], lut);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LUT_H_
