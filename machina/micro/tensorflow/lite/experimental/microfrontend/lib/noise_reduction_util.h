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
#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_

#include "machina/lite/experimental/microfrontend/lib/noise_reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NoiseReductionConfig {
  // scale the signal up by 2^(smoothing_bits) before reduction
  int smoothing_bits;
  // smoothing coefficient for even-numbered channels
  float even_smoothing;
  // smoothing coefficient for odd-numbered channels
  float odd_smoothing;
  // fraction of signal to preserve (1.0 disables this module)
  float min_signal_remaining;
};

// Populates the NoiseReductionConfig with "sane" default values.
void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config);

// Allocates any buffers.
int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels);

// Frees any allocated buffers.
void NoiseReductionFreeStateContents(struct NoiseReductionState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_
