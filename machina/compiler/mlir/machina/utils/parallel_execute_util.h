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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_PARALLEL_EXECUTE_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_PARALLEL_EXECUTE_UTIL_H_

#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"

namespace mlir {
namespace TF {

// TODO(b/243076653): Once the ParallelExecute is added do not remove it. This
//   means BuildSingletonParallelExecuteOp will be used in one location, and
//   RemoveSingletonParallelExecuteOp can be removed.

// Wrap `cluster_func` in a `ParallelExecute` with only one child. This
// can be used to canonicalize IR, so there is always one `ParallelExecute`.
tf_device::ParallelExecuteOp BuildParallelExecuteOp(
    tf_device::ClusterFuncOp cluster_func, OpBuilder* builder);

// Unwrap `parallel_execute`'s contents if it only has one child.
LogicalResult RemoveSingletonParallelExecuteOp(
    tf_device::ParallelExecuteOp parallel_execute, OpBuilder* builder);

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_PARALLEL_EXECUTE_UTIL_H_
