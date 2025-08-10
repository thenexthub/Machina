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

#ifndef MACHINA_COMPILER_MLIR_TFR_PASSES_PASSES_H_
#define MACHINA_COMPILER_MLIR_TFR_PASSES_PASSES_H_

#include <memory>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace mlir {
namespace TFR {

// Scans the func op and adds all the canonicalization patterns of the ops
// except the tf ops, inside the function.
void populateCanonicalizationPatterns(func::FuncOp func,
                                      RewritePatternSet &patterns);

// Decompose ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDecomposeTFOpsPass(
    std::optional<ModuleOp> tfr_module = std::nullopt);

// Rewrites quantized operands and results with their storage types.
// This pass should be run at module level after decomposition, if there are
// quantized operands or results.
std::unique_ptr<OperationPass<ModuleOp>> CreateRewriteQuantizedIOPass();

// Raise to TF ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseToTFOpsPass(
    std::optional<ModuleOp> tfr_module = std::nullopt,
    bool materialize_derived_attrs = false);

}  // namespace TFR
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TFR_PASSES_PASSES_H_
