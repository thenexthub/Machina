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
#ifndef MACHINA_LITE_MICRO_KERNELS_QUANTIZE_H_
#define MACHINA_LITE_MICRO_KERNELS_QUANTIZE_H_

#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/types.h"

namespace tflite {

struct OpDataQuantizeReference {
  tflite::QuantizationParams quantization_params;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t requantize_output_multiplier;
  int requantize_output_shift;

  int32_t input_zero_point;
};

TfLiteStatus EvalQuantizeReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus PrepareQuantizeReference(TfLiteContext* context, TfLiteNode* node);
}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_QUANTIZE_H_
