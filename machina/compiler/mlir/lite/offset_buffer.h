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
#ifndef MACHINA_COMPILER_MLIR_LITE_OFFSET_BUFFER_H_
#define MACHINA_COMPILER_MLIR_LITE_OFFSET_BUFFER_H_

#include <cstdint>

namespace tflite {

// Check if the model is using custom_option_offset to store custom op
// buffers. When this field is not explicitly set by the user, then FlatBuffer
// will omit the field and interpret this as 0, to ensure this field is
// populated. The flatbuffer exporter will always set it to 1, and it's also not
// a valid buffer offset value. So it's only valid when it's > 1.
inline bool IsValidBufferOffset(const int64_t offset) { return offset > 1; }

}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_OFFSET_BUFFER_H_
