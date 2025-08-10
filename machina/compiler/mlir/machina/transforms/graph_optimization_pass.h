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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_

#include <string>

#include "absl/status/status.h"
#include "machina/compiler/mlir/mlir_graph_optimization_pass.h"
#include "machina/core/protobuf/config.pb.h"

namespace mlir {
namespace TF {

// Bundle generic MLIR graph optimization passes (some derived from TF Grappler
// graph optimizers) into a single MLIR optimization pass.
class MlirGraphOptimizationPass : public ::machina::MlirOptimizationPass {
 public:
  toolchain::StringRef name() const override { return "graph_optimization"; }

  ::machina::MlirOptimizationPassState GetPassState(
      const ::machina::DeviceSet* device_set,
      const ::machina::ConfigProto& config_proto,
      const machina::Graph& graph,
      const machina::FunctionLibraryDefinition& function_library)
      const override {
    return config_proto.experimental().enable_mlir_graph_optimization()
               ? machina::MlirOptimizationPassState::Enabled
               : machina::MlirOptimizationPassState::Disabled;
  }

  absl::Status Run(
      const std::string& function_name,
      const ::machina::ConfigProto& config_proto, ModuleOp module,
      const ::machina::Graph& graph,
      const machina::FunctionLibraryDefinition& function_library) override;
};

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
