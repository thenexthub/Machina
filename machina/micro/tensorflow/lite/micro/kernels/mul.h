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

#ifndef MACHINA_LITE_MICRO_KERNELS_MUL_H_
#define MACHINA_LITE_MICRO_KERNELS_MUL_H_

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/micro/micro_common.h"

namespace tflite {

extern const int kMulInput1Tensor;
extern const int kMulInput2Tensor;
extern const int kMulOutputTensor;

struct OpDataMul {
  int32_t input1_zero_point;
  int32_t input2_zero_point;

  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_zero_point;
  int32_t output_multiplier;
  int output_shift;

  float output_activation_min_f32;
  float output_activation_max_f32;
};

void* MulInit(TfLiteContext* context, const char* buffer, size_t length);

TfLiteStatus CalculateOpDataMul(TfLiteContext* context, TfLiteNode* node,
                                TfLiteMulParams* params, OpDataMul* data);

TfLiteStatus MulPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus EvalMulQuantizedReference(TfLiteContext* context, TfLiteNode* node,
                                       const OpDataMul* data,
                                       const TfLiteEvalTensor* input1,
                                       const TfLiteEvalTensor* input2,
                                       TfLiteEvalTensor* output);

void EvalMulFloatReference(TfLiteContext* context, TfLiteNode* node,
                           TfLiteMulParams* params, const OpDataMul* data,
                           const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output);

// Generic must define registration function.
TFLMRegistration Register_MUL();

#if defined(CMSIS_NN)
TFLMRegistration Register_MUL_INT8();
#else
// Fallback registration
inline TFLMRegistration Register_MUL_INT8() { return Register_MUL(); }
#endif
}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_MUL_H_
