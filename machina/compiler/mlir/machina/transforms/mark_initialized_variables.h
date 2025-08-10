/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
// Marks all variables in 'function' whether they are initialized
// in 'session' or not by setting an attribute named 'is_initialized'
// on each variable op with value true/false based on variable is initialized
// in the session or not.
// If 'session' is NULL the function is no-op.
// Returns failure in case fetching variables from session failed, success
// otherwise.
LogicalResult MarkInitializedVariablesInFunction(func::FuncOp function,
                                                 machina::Session* session);
// Apply `MarkInitializedVariablesInFunction` to every non-empty function in the
// module.
LogicalResult MarkInitializedVariablesInFunction(ModuleOp module,
                                                 machina::Session* session);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_
