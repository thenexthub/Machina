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

#include "machina/core/transforms/toposort/pass.h"

#include <memory>
#include <vector>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/iterator_range.h"
#include "mlir/IR/RegionKindInterface.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/ops.h"

namespace mlir {
namespace tfg {

#define GEN_PASS_DEF_TOPOSORT
#include "machina/core/transforms/passes.h.inc"

void SortTopologically(Block *block, TFGraphDialect *dialect) {
  if (block->empty() || toolchain::hasSingleElement(*block)) return;

  Block::iterator first(&block->front());
  Block::iterator last = block->end();
  if (!block->getParentOp()->hasTrait<OpTrait::NoTerminator>()) --last;

  // Sort using a greedy worklist algorithm.
  std::vector<Operation *> worklist;
  for (Operation &op : toolchain::reverse(toolchain::make_range(first, last)))
    worklist.push_back(&op);

  // Track the scheduled ops.
  DenseSet<Operation *> scheduled;
  scheduled.reserve(worklist.size());

  // Given a top-level operation `top`, which is an operation in the block being
  // sorted, and an operand at or nested below the top-level operation, return
  // true if the operand is ready to be scheduled.
  const auto is_ready = [block, &scheduled](Value value, Operation *top) {
    // The given operand is ready if:
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (!parent) return true;
    Operation *ancestor = block->findAncestorOpInBlock(*parent);
    // - it is an implicit capture,
    if (!ancestor) return true;
    // - it is defined in a nested region, or
    if (ancestor == top) return true;
    // - its ancestor in the block is scheduled.
    return scheduled.contains(ancestor);
  };

  // An operation can be recursively scheduled if its and all its nested
  // operations' operands are ready.
  const auto can_be_scheduled = [&is_ready, dialect](Operation *op) {
    if (!op->getNumRegions()) {
      return toolchain::all_of(op->getOperands(), [&](Value operand) {
        if (is_ready(operand, op)) return true;
        // The only normally allowed cycles in TensorFlow graphs are backedges
        // from NextIteration to Merge nodes. Virtually break these edges by
        // marking them as ready.
        if (!dialect) return false;
        Operation *parent = operand.getDefiningOp();
        return parent && dialect->IsMerge(op) &&
               dialect->IsNextIteration(parent);
      });
    }
    return !op->walk([&](Operation *child) {
                return toolchain::all_of(
                           child->getOperands(),
                           [&](Value operand) { return is_ready(operand, op); })
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              }).wasInterrupted();
  };

  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();

    // Skip ops that are already scheduled.
    if (scheduled.contains(op)) continue;

    if (!can_be_scheduled(op)) continue;
    scheduled.insert(op);
    op->moveBefore(block, last);

    for (Operation *user : op->getUsers()) {
      // Push users to the front of the queue to optimistically schedule them
      // and maintain a relatively stable node order.
      worklist.push_back(block->findAncestorOpInBlock(*user));
    }
  }
}

namespace {

// A pass that topologically sort Graph regions.
struct TopoSortPass : impl::TopoSortBase<TopoSortPass> {
  void runOnOperation() override {
    auto *dialect = getContext().getLoadedDialect<TFGraphDialect>();

    Operation *op = getOperation();
    auto region_op = dyn_cast<RegionKindInterface>(op);
    if (!region_op) return;
    for (int region : toolchain::seq<int>(0, op->getNumRegions()))
      if (!region_op.hasSSADominance(region) && !op->getRegion(region).empty())
        SortTopologically(&op->getRegion(region).front(), dialect);
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTopoSortPass() {
  return std::make_unique<TopoSortPass>();
}

}  // namespace tfg
}  // namespace mlir
