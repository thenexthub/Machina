/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
// This file is the MLIR copy of part of
// third_party/machina/lite/kernels/internal/reference/portable_tensor_utils.h
// as part of the effort to decouple TFLite from MLIR.

#ifndef MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_PORTABLE_TENSOR_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_PORTABLE_TENSOR_UTILS_H_

#include <cstdint>

namespace mlir {
namespace lite {
namespace toco_legacy {

// LINT.IfChange(portable_symmetric_quantize_floats)
void PortableSymmetricQuantizeFloats(const float* values, int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor);

void PortableSymmetricQuantizeFloats(const float* values, int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor);
// LINT.ThenChange(//machina/lite/kernels/internal/reference/portable_tensor_utils.h:portable_symmetric_quantize_floats)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_PORTABLE_TENSOR_UTILS_H_
