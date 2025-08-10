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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/ops.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"

namespace mlir {
namespace tfg {

// Get the name of a value as if it were an edge in a graph.
absl::StatusOr<std::string> GetValueName(Value value, TFGraphDialect *dialect);

// Convert a TFG graph directly to GraphDef. Graph functions in the module are
// added to the GraphDef's function library.
absl::Status ConvertToGraphDef(ModuleOp module, machina::GraphDef *graph);

// Convert a single TFG op to NodeDef. This utliity function requires a callback
// `get_value_name` that returns the edge name of the given operand.
absl::Status ConvertToNodeDef(
    Operation *op, machina::NodeDef *node, TFGraphDialect *dialect,
    function_ref<absl::StatusOr<std::string>(Value)> get_value_name);

// Convert a single TFG function to a FunctionDef and add it to the function
// library. If a function with the same name already exists, replace it.
absl::Status ConvertToFunctionDef(
    GraphFuncOp func, machina::FunctionLibraryDefinition &library);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
