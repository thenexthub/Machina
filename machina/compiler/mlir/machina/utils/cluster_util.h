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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_CLUSTER_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_CLUSTER_UTIL_H_

#include <functional>
#include <string>

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/side_effect_analysis.h"

namespace mlir::TF {

// Cluster structure captures all the operations that are assigned to same
// device and can form a legal strict cluster.
// Ops must follow same ordering in their parent block. We rely on this
// assumption to perform analysis.
struct Cluster {
  toolchain::SetVector<Operation*> ops;
  std::string target;
};

// Builds the op clusters in the `block`. Ops are filtered by the function
// `get_target` that takes an op and returns the target name. `is_ignored_op` is
// a hook to ignore certain ops that are not included in any clusters.
toolchain::StringMap<SmallVector<Cluster>> BuildAllClusters(
    Block& block, const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    std::function<bool(Operation*)> is_ignored_op);

// Reorder all users of the given op's results to after the op.
//
// Since launch ops are inserted after the last op in the region, the region is
// guaranteed to dominate all live-in values. On the other hand, it is still
// possible that live-out values don't dominate the region. For example:
//
// ```
// %0 = "tf.OpA"()
// %1 = "tf.OpB"(%0)
// %2 = "tf.OpC"(%0)
// ```
//
// Assuming `tf.OpA` and `tf.OpC` are clustered together, the region will be
// inserted right after `tf.OpC`. The live-out `%0`, however, is used by
// `tf.OpB`, which won't dominate the region. This function reorders all users
// of the cluster op to be placed after the cluster op itself so that SSA
// dominance is preserved after cluster op creation.
void ReorderOpResultUses(mlir::Operation* cluster);

}  // namespace mlir::TF

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_CLUSTER_UTIL_H_
