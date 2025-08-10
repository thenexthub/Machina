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

#ifndef MACHINA_LITE_MICRO_KERNELS_LEAKY_RELU_H_
#define MACHINA_LITE_MICRO_KERNELS_LEAKY_RELU_H_

#include "machina/lite/c/common.h"

namespace tflite {

// Input/output tensor index.
extern const int kInputTensor;
extern const int kOutputTensor;

struct LeakyReluOpData {
  // quantization parameters
  int32_t output_multiplier_alpha;
  int32_t output_shift_alpha;
  int32_t output_multiplier_identity;
  int32_t output_shift_identity;
  int32_t input_zero_point;
  int32_t output_zero_point;
};

TfLiteStatus CalculateOpDataLeakyRelu(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus LeakyReluPrepare(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_LEAKY_RELU_H_
