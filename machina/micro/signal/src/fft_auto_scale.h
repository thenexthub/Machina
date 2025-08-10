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

#ifndef SIGNAL_SRC_FFT_AUTO_SCALE_H_
#define SIGNAL_SRC_FFT_AUTO_SCALE_H_

#include <stddef.h>
#include <stdint.h>

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflite {
namespace tflm_signal {

// Auto scales `input` and write the result to `output`
// Elements in `input` are left shifted to maximize the amplitude without
// clipping,
// * both `input` and `output` must be of size `size`
int FftAutoScale(const int16_t* input, int size, int16_t* output);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_FFT_AUTO_SCALE_H_
