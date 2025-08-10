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

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_

#include <memory>

#define GEN_PASS_DECL
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir::quant::stablehlo {

// Creates an instance of the ConvertTFQuantOpsToMHLOPass pass, which will
// convert TF uniform quantized ops to the corresponding quantized MHLO ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass();

// TODO(b/288094093): Migrate uniform quantization legalization in a separate
// pass.
void PopulateLegalizeTfQuantizationPatterns(MLIRContext *context,
                                            RewritePatternSet *patterns);

// Creates an instance of the ConvertTFQuantTypes pass, which will convert TF
// qint types to int types and surround TF UniformQuantized ops with qint <->
// int casts.
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass();

// Creates an instance of the VerifyQuantLegalization pass, which verifies all
// quant ops and types are lowered.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateVerifyQuantLegalizationPass();

// Add all passes for lowering TF quant ops and types to MHLO int.
void AddQuantizationLoweringPasses(mlir::OpPassManager &pm);

// Creates an instance of OptimizeIntGraphPass, which optimizes the int graph
// lowered from the quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeIntGraphPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CONVERTTFQUANTOPSTOMHLO
#define GEN_PASS_DECL_CONVERTTFQUANTTYPES
#define GEN_PASS_DECL_VERIFYQUANTLEGALIZATION
#define GEN_PASS_DECL_OPTIMIZEINTGRAPH
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"
}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_
