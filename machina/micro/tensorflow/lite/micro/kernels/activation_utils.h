/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
#define MACHINA_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_

#include <algorithm>
#include <cmath>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/kernels/internal/cppmath.h"
#include "machina/lite/kernels/internal/max.h"
#include "machina/lite/kernels/internal/min.h"

namespace tflite {
namespace ops {
namespace micro {

// Returns the floating point value for a fused activation:
inline float ActivationValFloat(TfLiteFusedActivation act, float a) {
  switch (act) {
    case kTfLiteActNone:
      return a;
    case kTfLiteActRelu:
      return TfLiteMax(0.0f, a);
    case kTfLiteActReluN1To1:
      return TfLiteMax(-1.0f, TfLiteMin(a, 1.0f));
    case kTfLiteActRelu6:
      return TfLiteMax(0.0f, TfLiteMin(a, 6.0f));
    case kTfLiteActTanh:
      return std::tanh(a);
    case kTfLiteActSignBit:
      return std::signbit(a);
    case kTfLiteActSigmoid:
      return 1.0f / (1.0f + std::exp(-a));
  }
  return 0.0f;  // To indicate an unsupported activation (i.e. when a new fused
                // activation is added to the enum and not handled here).
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
