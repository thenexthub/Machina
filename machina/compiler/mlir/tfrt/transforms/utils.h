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
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain

namespace machina {

// Checks if the given `value` is a resource argument.
bool IsResourceArgument(mlir::Value value);

// Checks if an operand is the value of a variable.
bool IsResultVariable(const mlir::Value &original_operand,
                      const mlir::Value &operand);

// Canonicalize the symbol attr to the original TF function name.
std::optional<std::string> CanonicalizeTensorflowFunctionName(
    const mlir::SymbolTable &symbol_table, absl::string_view mlir_func_name,
    bool use_mlir_func_name = false);

// Returns true if the function is a session initializer in tf_saved_model
// dialect.
bool IsSessionInitializer(mlir::func::FuncOp op);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_
