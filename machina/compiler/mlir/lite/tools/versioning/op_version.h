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
#ifndef MACHINA_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_VERSION_H_
#define MACHINA_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_VERSION_H_

#include <cstdint>

#include "machina/compiler/mlir/lite/schema/mutable/schema_generated.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/tools/versioning/op_signature.h"

namespace tflite {

// Returns version of builtin ops by the given signature.
int GetBuiltinOperatorVersion(const OpSignature& op_sig);

// Update operator's version of the given TFL flatbuffer model.
void UpdateOpVersion(uint8_t* model_buffer_pointer);

}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_VERSION_H_
