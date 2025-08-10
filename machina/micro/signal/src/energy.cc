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
#include "signal/src/energy.h"

#include "signal/src/complex.h"

namespace tflite {
namespace tflm_signal {
void SpectrumToEnergy(const Complex<int16_t>* input, int start_index,
                      int end_index, uint32_t* output) {
  for (int i = start_index; i < end_index; i++) {
    const int16_t real = input[i].real;  // 15 bits
    const int16_t imag = input[i].imag;  // 15 bits
    // 31 bits
    output[i] = (static_cast<int32_t>(real) * real) +
                (static_cast<int32_t>(imag) * imag);
  }
}

}  // namespace tflm_signal
}  // namespace tflite
