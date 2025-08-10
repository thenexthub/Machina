/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_

#include <stdint.h>
#include <stdlib.h>

#define kPcanSnrBits 12
#define kPcanOutputBits 6

#ifdef __cplusplus
extern "C" {
#endif

// Details at https://research.google/pubs/pub45911.pdf
struct PcanGainControlState {
  int enable_pcan;
  uint32_t* noise_estimate;
  int num_channels;
  int16_t* gain_lut;
  int32_t snr_shift;
};

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

uint32_t PcanShrink(const uint32_t x);

void PcanGainControlApply(struct PcanGainControlState* state, uint32_t* signal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_
