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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_TOPOLOGICAL_SORT_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_TOPOLOGICAL_SORT_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "toolchain/ADT/STLFunctionalExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain

namespace mlir {
namespace TF {

// A function that determines which op to emit next in the case of ties.
// The predecessor (which can be null) is the last op we emitted,
// and op is the candidate we're considering. A larger returned integer
// means the op has a higher chance of being emitted first.
typedef int (*PriorityFunction)(Operation *predecessor, Operation *op);

// A function that returns extra dependencies for each op. These might
// e.g. be known side-effects (or control dependencies) between ops.
// If "incoming" is true, then the list of (extra) predecessors of the
// op should be returned. If "incoming" is false, the list of successors.
// The algorithm assumes that these are consistent which each other. So
// if (and only if) op1 is in extra_dependencies(op2, true), then op2
// must also be in extra_dependencies(op1, false).
// This function is called multiple times during the topological sort,
// so the implementation should preferably be constant-time.
typedef toolchain::function_ref<toolchain::SmallVector<Operation *, 4> const &(
    Operation *, bool incoming)>
    ExtraDependenciesFunction;

// Convenience function if there are no extra dependencies to declare.
// (Unlike nullptr, this also works inside the ternary operator)
extern ExtraDependenciesFunction no_extra_dependencies;

// Sort a block topologically, so that for all ops, all operands are
// available at the time of execution.  This is similar to MLIR's topological
// sort (lib/Transforms/TopologicalSort.cpp) but also takes a priority
// function to determine the next op to emit in the case of ambiguity. This
// makes it possible to group operations by certain attributes. For example,
// the order_by_dialect pass uses this function to group by dialect.
// Only the operations nested directly under the block will be reordered.
// Nested blocks will be left alone.
// Also takes a list of control dependencies (vector of operation pairs,
// from->to) that will be honored when ordering the ops together with the
// data dependencies given through (the ops/results of) the operations
// themselves.
std::vector<Operation *> SortBlockTopologically(
    Block &block, PriorityFunction priorityFunction,
    ExtraDependenciesFunction extraDependencies = no_extra_dependencies);

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_TOPOLOGICAL_SORT_H_
