/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_
#define MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_

#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain

namespace mlir::odml {

// Patterns to legalize mhlo.reduce_window to TFL.
//
// Maps the following representations of AvgPool in MHLO into a tfl.avg_pool
// operation when they cleanly map to 2D or 3D average pool with VALID or SAME
// padding:
// * div(reduce_sum_window(x), constant(sizeof(window)))
// * div(reduce_sum_window(x), reduce_sum_window(constant(1)))
//
// Emits: tfl.average_pool2d
void PopulateLegalizeReduceWindowPatterns(MLIRContext* ctx,
                                          RewritePatternSet& patterns,
                                          ConversionTarget& target);

// Patterns to prepare mhlo.reduce_window for legalization.
// Transposes reduce_windows to be NHWC.
//
// Emits: tfl.transpose
void PopulatePrepareReduceWindowPatterns(MLIRContext* ctx,
                                         RewritePatternSet& patterns);

}  // namespace mlir::odml

#endif  // MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_H_
