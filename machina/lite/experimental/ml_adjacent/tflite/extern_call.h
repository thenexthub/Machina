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
#ifndef MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_
#define MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_

#include <cstdint>

#include "machina/lite/core/c/common.h"

namespace tflite::extern_call {

// Compile time options passed to this kernel at runtime.
struct ExternCallOptions {
  // A single custom op is used to represent a call to an arbitrary function
  // in the library. The function that is called is encoded in `func_id`.
  // Because of compiler op def, these will be encded at compile time
  // as `char[]` and will serialize `uint8_t`, so we match this type.
  uint8_t func_id;
};

TfLiteRegistration* Register_EXTERN_CALL();

}  // namespace tflite::extern_call

#endif  // MACHINA_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_
