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

#include "machina/core/transforms/graph_transform_wrapper.h"

#include <initializer_list>
#include <memory>

#include "absl/status/status.h"
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/graph/graph.h"
#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/ir/importexport/graphdef_import.h"
#include "machina/core/platform/statusor.h"

namespace mlir {
namespace tfg {

absl::Status RunTransformOnGraph(
    machina::Graph* graph,
    const std::initializer_list<
        toolchain::function_ref<std::unique_ptr<mlir::Pass>()>>& passes,
    const machina::GraphDebugInfo& debug_info) {
  // We are running only a set of Module passes on a Modul, so disable threading
  // to avoid overhead of creating threadpool that won't be used.
  MLIRContext context(MLIRContext::Threading::DISABLED);
  TF_ASSIGN_OR_RETURN(OwningOpRef<ModuleOp> module,
                      ImportGraphAndFunctionsToMlir(&context, debug_info,
                                                    *graph, graph->flib_def()));

  PassManager pm((*module)->getName(), mlir::PassManager::Nesting::Explicit);
  // Construct passes.
  for (auto& pass : passes) pm.addPass(pass());
  mlir::StatusScopedDiagnosticHandler error_handler(&context);
  if (failed(pm.run(*module)))
    return error_handler.Combine(
        machina::errors::InvalidArgument("MLIR Graph Optimizer failed: "));

  // Export and replace Graph.
  machina::GraphDef graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertToGraphDef(*module, &graphdef),
                                  "when exporting MLIR module to GraphDef");
  graph->Clear();
  graph->mutable_flib_def()->Clear();
  machina::GraphConstructorOptions opts;
  return ConvertGraphDefToGraph(opts, graphdef, graph);
}

}  // namespace tfg
}  // namespace mlir
