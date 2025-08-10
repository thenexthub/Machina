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
#ifndef MACHINA_LITE_MICRO_KERNELS_SOFTMAX_H_
#define MACHINA_LITE_MICRO_KERNELS_SOFTMAX_H_

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/micro/micro_common.h"

namespace tflite {

void* SoftmaxInit(TfLiteContext* context, const char* buffer, size_t length);

// Common helper function to SoftmaxPrepare.
TfLiteStatus CalculateSoftmaxParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data);

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node);

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_SOFTMAX();

#if defined(XTENSA) || defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 input and int16 output.
TFLMRegistration Register_SOFTMAX_INT8_INT16();
#else
inline TFLMRegistration Register_SOFTMAX_INT8_INT16() {
  return Register_SOFTMAX();
}
#endif

#if defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 input/output and uses the latency optimized implementations.
TFLMRegistration Register_SOFTMAX_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 input/output and uses the latency optimized implementations.
TFLMRegistration Register_SOFTMAX_INT16();

#else
inline TFLMRegistration Register_SOFTMAX_INT8() { return Register_SOFTMAX(); }

inline TFLMRegistration Register_SOFTMAX_INT16() { return Register_SOFTMAX(); }
#endif

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_SOFTMAX_H_
