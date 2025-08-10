/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/lite/core/c/common.h"
#include "machina/lite/kernels/stablehlo_elementwise.h"

namespace tflite::ops::builtin {

TfLiteRegistration* Register_STABLEHLO_MAXIMUM() {
  static TfLiteRegistration r = {nullptr, nullptr, ElementwisePrepare,
                                 ElementwiseEval<ComputationType::kMax>};
  return &r;
}
TfLiteRegistration* Register_STABLEHLO_MINIMUM() {
  static TfLiteRegistration r = {nullptr, nullptr, ElementwisePrepare,
                                 ElementwiseEval<ComputationType::kMin>};
  return &r;
}
}  // namespace tflite::ops::builtin
