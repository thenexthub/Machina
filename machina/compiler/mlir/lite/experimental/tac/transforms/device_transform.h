/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {

// Returns true if 'op' is supported to run on 'hardware'.
bool IsSupported(Operation* op, const std::string& hardware);

// Return proper rewriter patterns for different hardwares.
RewritePatternSet GetHardwareRewritePatterns(MLIRContext* context,
                                             const std::string& hardware);

// Convert quantized ops to float, this will essentially insert dequantize &
// quantize pair around the op.
void ConvertQuantizedOpToFloat(func::FuncOp func, OpBuilder* builder);

// This will optimize the quantized ops -> float graph.
void OptimizeQuantizedOpToFloat(func::FuncOp func, MLIRContext* context);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_
