/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_

#include <memory>

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir {
namespace TFTPU {

// Creates a pass that rewrites `tf_device.launch_func` on TPUs into TPU runtime
// ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateTPURewritePass(
    toolchain::StringRef module_name = toolchain::StringRef());

// Creates a pass that adds ops which perform formatting on variables at
// run-time according to compilation result.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUVariableRuntimeReformattingPass();

// Creates a pass that merges device variable reads/updates into the surrounded
// TPUExecute node. This allows the execute node to perform in-place variable
// updates.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUMergeVariablesWithExecutePass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_TPUMERGEVARIABLESWITHEXECUTEPASS
#define GEN_PASS_DECL_TPUREWRITEPASS
#define GEN_PASS_DECL_TPUVARIABLERUNTIMEREFORMATTINGPASS
#include "machina/compiler/mlir/machina/transforms/host_runtime/runtime_passes.h.inc"

}  // namespace TFTPU
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_
