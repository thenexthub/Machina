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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LIFT_TFLITE_FLEX_OPS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LIFT_TFLITE_FLEX_OPS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {

// Creates an instance of the lift TFLite Flex ops pass that lifts TFLite Flex
// ops into TF dialect operations.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLiftTfliteFlexOpsPass();

void AddLiftTfliteFlexOpsPatterns(MLIRContext *context,
                                  RewritePatternSet &patterns);

}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_LIFT_TFLITE_FLEX_OPS_H_
