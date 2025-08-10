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
#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_

#include <stdint.h>
#include <stdlib.h>

#include "machina/lite/experimental/microfrontend/lib/fft.h"
#include "machina/lite/experimental/microfrontend/lib/filterbank.h"
#include "machina/lite/experimental/microfrontend/lib/log_scale.h"
#include "machina/lite/experimental/microfrontend/lib/noise_reduction.h"
#include "machina/lite/experimental/microfrontend/lib/pcan_gain_control.h"
#include "machina/lite/experimental/microfrontend/lib/window.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FrontendState {
  struct WindowState window;
  struct FftState fft;
  struct FilterbankState filterbank;
  struct NoiseReductionState noise_reduction;
  struct PcanGainControlState pcan_gain_control;
  struct LogScaleState log_scale;
};

struct FrontendOutput {
  const uint16_t* values;
  size_t size;
};

// Main entry point to processing frontend samples. Updates num_samples_read to
// contain the number of samples that have been consumed from the input array.
// Returns a struct containing the generated output. If not enough samples were
// added to generate a feature vector, the returned size will be 0 and the
// values pointer will be NULL. Note that the output pointer will be invalidated
// as soon as FrontendProcessSamples is called again, so copy the contents
// elsewhere if you need to use them later.
struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read);

void FrontendReset(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_
