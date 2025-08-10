/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_SAVE_VARIABLES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_SAVE_VARIABLES_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain

namespace machina {
namespace quantization {

// Saves variables in `module_op` to the checkpoint file inside `prefix`.
// It finds variables that are initialized with "tf.AssignVariableOp" inside the
// initializer function with type "restore_op". The "tf.Const"s used to
// initialize the variables are saved. This function does not modify the
// `module_op`. Returns a list of saved names of the saved variables.
absl::StatusOr<std::vector<std::string>> SaveVariablesToCheckpoint(
    absl::string_view prefix, mlir::ModuleOp module_op);

}  // namespace quantization
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_SAVE_VARIABLES_H_
