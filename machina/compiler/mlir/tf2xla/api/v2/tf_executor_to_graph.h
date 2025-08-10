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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_EXECUTOR_TO_GRAPH_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_EXECUTOR_TO_GRAPH_H_

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/graph/graph.h"

namespace machina {
namespace tf2xla {
namespace v2 {

// Converts an MLIR module to TensorFlow graph and FunctionLibraryDefinition.
// The "main" function of the module is stored in the graph and the rest of
// functions are stored in the library. Control ret nodes are stored separately
// in `control_ret_nodes`.
absl::Status ConvertTfExecutorToGraph(
    mlir::ModuleOp module, const GraphExportConfig& configs,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    absl::flat_hash_set<Node*>* control_ret_nodes);

// Converts an MLIR function and adds it to a FunctionLibraryDefinition.
absl::Status ConvertMlirFunctionToFunctionLibraryDef(
    mlir::func::FuncOp func, const GraphExportConfig& configs,
    FunctionDef* function_def);

}  // namespace v2
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_EXECUTOR_TO_GRAPH_H_
