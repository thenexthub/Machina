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

#ifndef MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_
#define MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir {
namespace odml {

// Applies various optimizations on MHLO IR.
std::unique_ptr<Pass> createOptimizePass();

// Finds quantization patterns and compose them to uniform
// quantized types.
std::unique_ptr<OperationPass<ModuleOp>>
CreateComposeUniformQuantizedTypePass();

// Finds stablehlo ops that accept or produce uniform
// quantized typed tensors and converts them to equivalent ops in the TFLite
// dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateUniformQuantizedStableHloToTflPass();

// Commutes transposes through specific ops
std::unique_ptr<OperationPass<ModuleOp>> CreateTransposeCommuteOpsPass();

// Legalizes MHLO to TF dialect.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfPass();

// Replaces a splat constant tensor with a BroadcastInDim
// op.
std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldSplatConstantPass();

// Legalizes MHLO to TFLite dialect.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfLitePass();

// Lowers stablehlo composite ops to tflite ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateCompositeLoweringPass();

// Legalizes CHLO to tflite dialect.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeChloToTflPass();

// Rewrites MHLO in preparation for tflite legalization.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareHloPass();

// Adds the HLO to TF rewrite patterns to the specified pattern list.
void PopulateLegalizeHloToTfPatterns(RewritePatternSet* patterns,
                                     MLIRContext* context);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

}  // namespace odml
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_
