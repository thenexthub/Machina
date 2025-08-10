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
#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_

#include "machina/lite/experimental/microfrontend/lib/fft_util.h"
#include "machina/lite/experimental/microfrontend/lib/filterbank_util.h"
#include "machina/lite/experimental/microfrontend/lib/frontend.h"
#include "machina/lite/experimental/microfrontend/lib/log_scale_util.h"
#include "machina/lite/experimental/microfrontend/lib/noise_reduction_util.h"
#include "machina/lite/experimental/microfrontend/lib/pcan_gain_control_util.h"
#include "machina/lite/experimental/microfrontend/lib/window_util.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FrontendConfig {
  struct WindowConfig window;
  struct FilterbankConfig filterbank;
  struct NoiseReductionConfig noise_reduction;
  struct PcanGainControlConfig pcan_gain_control;
  struct LogScaleConfig log_scale;
};

// Fills the frontendConfig with "sane" defaults.
void FrontendFillConfigWithDefaults(struct FrontendConfig* config);

// Allocates any buffers.
int FrontendPopulateState(const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate);

// Frees any allocated buffers.
void FrontendFreeStateContents(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_
