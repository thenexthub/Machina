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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_SESSION_UTILS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_SESSION_UTILS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "toolchain/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

// Returns the variable for the provided 'var_handle_op'.
std::string GetVariableName(TF::VarHandleOp var_handle_op);

// Returns pointer to the variable from 'session' that 'var_handle_op'
// refers to which is in 'device_name' device. If failed to fetch the value null
// will be returned.
// Note, caller is responsible for Unref the variable.
machina::Var* GetVariableFromSession(mlir::TF::VarHandleOp var_handle_op,
                                        toolchain::StringRef device_name,
                                        const machina::DeviceMgr* mgr);

// Returns resource tensors from session for all variables in 'module'.
absl::StatusOr<std::vector<machina::Tensor>> GetResourcesFromSession(
    toolchain::ArrayRef<TF::VarHandleOp> var_handle_ops,
    machina::Session* session);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_SESSION_UTILS_H_
