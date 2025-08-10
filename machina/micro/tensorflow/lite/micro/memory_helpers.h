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
#ifndef MACHINA_LITE_MICRO_MEMORY_HELPERS_H_
#define MACHINA_LITE_MICRO_MEMORY_HELPERS_H_

#include <cstddef>
#include <cstdint>

#include "machina/lite/c/common.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {

// Returns the next pointer address aligned to the given alignment.
uint8_t* AlignPointerUp(uint8_t* data, size_t alignment);

// Returns the previous pointer address aligned to the given alignment.
uint8_t* AlignPointerDown(uint8_t* data, size_t alignment);

// Returns an increased size that's a multiple of alignment.
size_t AlignSizeUp(size_t size, size_t alignment);

// Templated version of AlignSizeUp
// Returns an increased size that's a multiple of alignment.
template <typename T>
size_t AlignSizeUp(size_t count = 1) {
  return AlignSizeUp(sizeof(T) * count, alignof(T));
}

// Returns size in bytes for a given TfLiteType.
TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size);

// How many bytes are needed to hold a tensor's contents.
TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size);

// How many bytes are used in a TfLiteEvalTensor instance. The byte length is
// returned in out_bytes.
TfLiteStatus TfLiteEvalTensorByteLength(const TfLiteEvalTensor* eval_tensor,
                                        size_t* out_bytes);

// Deduce output dimensions from input and allocate given size.
// Useful for operators with two inputs where the largest input should equal the
// output dimension.
TfLiteStatus AllocateOutputDimensionsFromInput(TfLiteContext* context,
                                               const TfLiteTensor* input1,
                                               const TfLiteTensor* input2,
                                               TfLiteTensor* output);

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_MEMORY_HELPERS_H_
