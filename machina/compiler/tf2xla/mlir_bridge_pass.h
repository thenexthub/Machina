/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MLIR_BRIDGE_PASS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MLIR_BRIDGE_PASS_H_

#include <string>

#include "machina/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"
#include "absl/status/status.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/jit/flags.h"
#include "machina/compiler/mlir/mlir_graph_optimization_pass.h"
#include "machina/core/common_runtime/device_set.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow Function Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgePass : public MlirOptimizationPass {
 public:
  toolchain::StringRef name() const override { return "bridge"; }

  MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const override;

  // This should be used as a thin mapper around mlir::ModulePass::runOnModule
  // API integrated with the Tensorflow runtime.
  absl::Status Run(const std::string& function_name,
                   const ConfigProto& config_proto, mlir::ModuleOp module,
                   const Graph& graph,
                   const FunctionLibraryDefinition& function_library) override;
};

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow V1 Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgeV1CompatPass : public MlirV1CompatOptimizationPass {
 public:
  toolchain::StringRef name() const override { return "bridge"; }

  MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const override;

  // This should be used as a thin mapper around mlir::ModulePass::runOnModule
  // API integrated with the Tensorflow runtime.
  absl::Status Run(const GraphOptimizationPassOptions& options,
                   mlir::ModuleOp module) override;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MLIR_BRIDGE_PASS_H_
