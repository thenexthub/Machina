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

#include "signal/src/filter_bank_square_root.h"

#include "signal/src/square_root.h"

namespace tflite {
namespace tflm_signal {

void FilterbankSqrt(const uint64_t* input, int num_channels,
                    int scale_down_bits, uint32_t* output) {
  for (int i = 0; i < num_channels; ++i) {
    output[i] = Sqrt64(input[i]) >> scale_down_bits;
  }
}

}  // namespace tflm_signal
}  // namespace tflite
