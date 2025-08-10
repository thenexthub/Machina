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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_DECOMPOSE_RESOURCE_OPS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_DECOMPOSE_RESOURCE_OPS_H_

#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain

namespace mlir {
namespace TF {

// Populates rewrite patterns that decompose composite resource operations into
// primitive ones like ReadVariableOp, AssignVariableOp and other computations
// to facilitate transformations like resource op lifting.
// NOTE: These patterns do not support `use_locking=true` for a lot of resource
// operations. So decomposition may not be correct outside of backends like XLA,
// which automatically locks all resource variables.
void PopulateDecomposeResourceOpsPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns);

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_DECOMPOSE_RESOURCE_OPS_H_
