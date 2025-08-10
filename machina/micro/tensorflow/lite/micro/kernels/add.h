/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_LITE_MICRO_KERNELS_ADD_H_
#define MACHINA_LITE_MICRO_KERNELS_ADD_H_

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/micro/micro_common.h"

namespace tflite {

extern const int kAddInputTensor1;
extern const int kAddInputTensor2;
extern const int kAddOutputTensor;

struct OpDataAdd {
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32_t input1_multiplier;
  int32_t input2_multiplier;
  int32_t output_multiplier;
  int output_shift;
  int left_shift;
  int32_t input1_offset;
  int32_t input2_offset;
  int32_t output_offset;

  // Used only for float evals:
  float output_activation_min_f32;
  float output_activation_max_f32;
};

TfLiteStatus CalculateOpDataAdd(TfLiteContext* context, TfLiteAddParams* params,
                                const TfLiteTensor* input1,
                                const TfLiteTensor* input2,
                                TfLiteTensor* output, OpDataAdd* data);

TfLiteStatus AddPrepare(TfLiteContext* context, TfLiteNode* node);

// Generic must define registration function.
TFLMRegistration Register_ADD();

#if defined(CMSIS_NN)
TFLMRegistration Register_ADD_INT8();

TFLMRegistration Register_ADD_INT16();
#else
// Fallback registration
inline TFLMRegistration Register_ADD_INT8() { return Register_ADD(); }

inline TFLMRegistration Register_ADD_INT16() { return Register_ADD(); }
#endif
}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_ADD_H_
