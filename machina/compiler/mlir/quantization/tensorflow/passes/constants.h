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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PASSES_CONSTANTS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PASSES_CONSTANTS_H_

#include "toolchain/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace quant {

// Name of the save function. The "tf_quant__" prefix is for avoiding conflict
// with existing function's name.
inline constexpr StringRef kTfQuantSaveFuncName = "tf_quant__save";

// Name of the TensorFlow Operation to be fetched to save the variables to
// checkpoint. This save op follows the SavedModel's load semantics, so it
// should return the file prefix of the checkpoint as a string tensor.
inline constexpr StringRef kTfQuantSaveOpName = "tf_quant__save_op";

// Name the file prefix string tensor. The tensor is used to identify the prefix
// to the checkpoint where the variables are saved / loaded. This may be present
// in a function argument's "tf_saved_model.index_path" attribute to identify
// the file prefix function argument.
inline constexpr StringRef kTfFilePrefix = "__tf_file_prefix";

}  // namespace quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PASSES_CONSTANTS_H_
