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
#ifndef SIGNAL_SRC_OVERLAP_ADD_H_
#define SIGNAL_SRC_OVERLAP_ADD_H_

#include <stddef.h>
#include <stdint.h>

namespace tflm_signal {
// Adds (with saturation) the contents of `input` to the contents of `buffer`,
// both of size `input_size`, then copies the first `output_size` elements of
// `buffer` to `output`, shifts the last `input_size`-`output_size` elements of
// `buffer` to the beginning of `buffer` and fills the trailing `output_size`
// samples in `buffer` with zeros.
// input: {input[0] ... input[input_size-1]}
// buffer: {buffer[0] ... buffer[input_size-1]}
// After invocation:
// output: {saturate(input[0] + buffer[0]),
//          ... ,
//          saturate(input[output_size-1] + buffer[output_size-1])}
// buffer: {saturate(input[output_size] + buffer[output_size]),
//          ...
//          saturate(  input[input_size-output_size-1]
//                   + buffer[input_size-output_size-1]),
//          zeros(output_size)}
void OverlapAdd(const int16_t* input, int16_t* buffer, int input_size,
                int16_t* output, int output_size);

// The same as the int16_t variant above, but without saturation
void OverlapAdd(const float* input, float* buffer, int input_size,
                float* output, int output_size);

}  //  namespace tflm_signal
#endif  // SIGNAL_SRC_OVERLAP_ADD_H_
