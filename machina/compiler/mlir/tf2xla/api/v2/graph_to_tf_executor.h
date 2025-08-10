/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_GRAPH_TO_TF_EXECUTOR_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_GRAPH_TO_TF_EXECUTOR_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/graph/graph.h"

namespace machina {
namespace tf2xla {
namespace v2 {

inline constexpr absl::string_view kImportModelDefaultGraphFuncName = "main";

// Given a Graph, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToTfExecutor(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name =
        nullptr,
    const ConfigProto& config_proto = {},
    machina::TF2XLABridgeVersion bridge_version =
        machina::TF2XLABridgeVersion::kNotBridgeUseCase);

}  // namespace v2
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_GRAPH_TO_TF_EXECUTOR_H_
