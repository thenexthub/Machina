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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_LOWERING_PASSES_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_LOWERING_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace machina {
namespace tf2xla {
namespace internal {

// Create a pass that just collects metrics about the input MLIR. Does not
// logically transform the program.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInputLoweringMetricsPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_INPUTLOWERINGMETRICSPASS
#include "machina/compiler/mlir/tf2xla/internal/passes/lowering_passes.h.inc"

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_LOWERING_PASSES_H_
