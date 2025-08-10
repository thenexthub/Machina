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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_VERIFY_SUITABLE_FOR_GRAPH_EXPORT_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_VERIFY_SUITABLE_FOR_GRAPH_EXPORT_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace machina {

// Returns whether all functions in module are of single tf_executor.graph and
// each tf_executor.island in tf_executor.graph only has a single op.
mlir::LogicalResult VerifyExportSuitable(mlir::ModuleOp module);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_VERIFY_SUITABLE_FOR_GRAPH_EXPORT_H_
