/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_LITE_MICRO_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H_
#define MACHINA_LITE_MICRO_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H_

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/types.h"

namespace tflite {

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
// TODO(b/230666079): resolve conflict with xtensa implementation
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM();

#if defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT16();

#else
inline TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8() {
  return Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
}

inline TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT16() {
  return Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
}
#endif

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H_
