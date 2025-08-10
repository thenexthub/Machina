/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_LITE_MICRO_KERNELS_PAD_H_
#define MACHINA_LITE_MICRO_KERNELS_PAD_H_

#include "machina/lite/c/common.h"
#include "machina/lite/micro/kernels/kernel_util.h"

namespace tflite {

struct OpData {
  PadParams params;
  int32_t output_zero_point;
};

void* PadInit(TfLiteContext* context, const char* buffer, size_t length);
TfLiteStatus PadPrepare(TfLiteContext* context, TfLiteNode* node);

TFLMRegistration Register_PAD();
TFLMRegistration Register_PADV2();

#if defined(CMSIS_NN)
TFLMRegistration Register_PAD_INT8();
#else
inline TFLMRegistration Register_PAD_INT8() { return Register_PAD(); }
#endif

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_PAD_H_
