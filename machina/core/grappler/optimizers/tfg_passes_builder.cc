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

#include "machina/core/grappler/optimizers/tfg_passes_builder.h"

#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/core/ir/ops.h"
#include "machina/core/protobuf/rewriter_config.pb.h"
#include "machina/core/transforms/pass_registration.h"
#include "machina/core/util/util.h"

namespace mlir {
namespace tfg {

// The default pipeline only does shape inference now.
void DefaultGrapplerPipeline(PassManager& manager) {
  // Turn certain shape attrs into types to give better knowledge for shape
  // inference.
  manager.addPass(CreateConsolidateAttributesPass());
  // Toposort the graph will bring better performance in some optimizations like
  // shape inference.
  manager.addPass(CreateTopoSortPass());
  // Infer the shape of operation if possible. The TFG importer doesn't do shape
  // inference for almost all operations.
  manager.addPass(CreateShapeInferencePass());
  // Contruct the shape attrs back from types.
  manager.addPass(CreatePrepareAttributesForExportPass());
}

// Run the consolidate attributes pass. Convert the whole module to region
// control-flow and run control-flow sinking. Convert the whole module back to
// functional control-flow and prepare the attributes for export.
void DefaultModuleGrapplerPipeline(PassManager& manager,
                                   const machina::RewriterConfig& config) {
  manager.addPass(CreateConsolidateAttributesPass());
  manager.addPass(CreateFunctionalToRegionPass());
  if (config.experimental_conditional_code_motion() !=
      machina::RewriterConfig::OFF)
    manager.addNestedPass<GraphFuncOp>(CreateControlFlowSinkPass());
  manager.addPass(CreateRegionToFunctionalPass(/*force_control_capture=*/true));
  manager.addPass(CreateLiftLegacyCallPass());
  manager.addPass(createSymbolPrivatizePass());
  manager.addPass(createSymbolDCEPass());
  manager.addPass(CreatePrepareAttributesForExportPass());
}

void RemapperPassBuilder(PassManager& manager) {
  manager.addPass(CreateConsolidateAttributesPass());
  manager.addPass(CreateTopoSortPass());
  manager.addPass(CreateShapeInferencePass());
  manager.addPass(
      CreateRemapperPass(/*enable_onednn_patterns=*/machina::IsMKLEnabled(),
                         /*xla_auto_clustering=*/false));
  manager.addPass(CreatePrepareAttributesForExportPass());
}

}  // namespace tfg
}  // namespace mlir
