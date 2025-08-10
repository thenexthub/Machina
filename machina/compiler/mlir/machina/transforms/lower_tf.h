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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_LOWER_TF_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_LOWER_TF_H_

#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain

namespace mlir {
namespace TF {

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented using other TensorFlow operations.
// TODO(laurenzo): For some reason, TFLite uses this pass and has exact
// requirements on what it can do. This is fragile and should be fixed (at a
// minimum, names should clearly convey scope). In the mean time, for a real
// compiler, use PopulateTFLoweringBeforeHLOPatterns.
void PopulateLoweringTFPatterns(MLIRContext *context,
                                RewritePatternSet *patterns);

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented by means of other TensorFlow operations.
// This pattern collection preserves those TensorFlow operations that will later
// be lowered to equivalent operations in CHLO or MHLO. This allows for
// HLO-specific lowerings.
void PopulateTFLoweringBeforeHLOPatterns(MLIRContext *context,
                                         RewritePatternSet *patterns);

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented using other TensorFlow operations.
// Patterns are from ops with some inputs or outputs that are quantized types
// only to ops that allow non-quantized types on all inputs and outputs.
void PopulateLoweringQuantizedPatterns(MLIRContext *context,
                                       RewritePatternSet *patterns);

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_LOWER_TF_H_
