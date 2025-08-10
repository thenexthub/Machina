/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/compiler/mlir/machina/transforms/graph_optimization_pass.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/core/protobuf/config.pb.h"

namespace mlir {
namespace TF {
namespace {
using Status = absl::Status;
using ConfigProto = ::machina::ConfigProto;
using Graph = ::machina::Graph;
}  // namespace

Status MlirGraphOptimizationPass::Run(
    const std::string& function_name, const ConfigProto& config_proto,
    ModuleOp module, const Graph& graph,
    const machina::FunctionLibraryDefinition& function_library) {
  if (GetPassState(/*device_set=*/nullptr, config_proto, graph,
                   function_library) ==
      ::machina::MlirOptimizationPassState::Disabled) {
    VLOG(1) << "Skipping MLIR Graph Optimization Pass"
            << ", session flag not enabled";
    return absl::OkStatus();
  }

  VLOG(1) << "Run MLIR Graph Optimization Passes";
  PassManager pm(module.getContext());
  ::machina::applyTensorflowAndCLOptions(pm);

  // Run island coarsening before shape inference to allow more exact shape
  // inference using constant folding within islands.
  pm.addNestedPass<func::FuncOp>(
      tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(CreateTFShapeInferencePass());

  // Assign optimal data layout to layout sensitive operations and delete
  // redundant transposes from the IR.
  LayoutOptimizationPipelineOptions layout_optimization_options;
  CreateLayoutOptimizationPipeline(pm.nest<func::FuncOp>(),
                                   layout_optimization_options);

  // Prepare IR for exporting.
  pm.addPass(CreateBreakUpIslandsPass());

  // In case of failure, the `diag_handler` converts MLIR errors emitted to the
  // MLIRContext into a machina::Status.
  StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = pm.run(module);
  (void)result;
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir
