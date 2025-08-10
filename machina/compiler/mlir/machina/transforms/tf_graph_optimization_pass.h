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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/core/common_runtime/optimization_registry.h"

namespace machina {

// Create a module pass that will execute the given TF GraphOptimization passes
// in sequence.
// Pass requires that the module ran on is convertible to TF Graph.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTensorFlowGraphOptimizationPass(
    std::vector<machina::GraphOptimizationPass*> tf_passes);

// Same as above but pass names instead of the passes provided. The registered
// passes are queried, if a TF graph optimization pass is not found in registry
// then the pass fails.
// Pass requires that the module ran on is convertible to TF Graph.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTensorFlowGraphOptimizationPass(
    const std::vector<std::string>& pass_names);

// Register the pass for command line testing.
void RegisterGraphOptimizationPasses();

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_
