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

#include <memory>

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Region.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_remaining_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/topological_sort.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_MOVETPUCOMPILETOFRONTPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct MoveTpuCompileToFrontPass
    : public impl::MoveTpuCompileToFrontPassBase<MoveTpuCompileToFrontPass> {
  void runOnOperation() override;
};

void MarkCompilationOps(Operation* func) {
  func->walk([&](Operation* op) {
    if (toolchain::isa<TF::_TPUCompileMlirOp>(op)) {
      op->setAttr("_is_compilation", UnitAttr::get(func->getContext()));
      op = op->getParentOp();
      while (op && op != func) {
        op->setAttr("_wraps_compilation", UnitAttr::get(func->getContext()));
        op = op->getParentOp();
      }
    }
  });
}

void UnmarkCompilationOps(Operation* func) {
  func->walk([&](Operation* op) {
    while (op && op != func) {
      op->removeAttr("_is_compilation");
      op->removeAttr("_wraps_compilation");
      op = op->getParentOp();
    }
  });
}

int OutsideCompilationOrdering(Operation* predecessor, Operation* op) {
  // Actual compilations go first.
  if (op->hasAttr("_is_compilation")) return 2;
  // Followed by nested ops that contain compilations.
  if (op->hasAttr("_wraps_compilation")) return 1;
  // Followed by everything else.
  return 0;
}

void MoveTpuCompileToFrontPass::runOnOperation() {
  MarkCompilationOps(getOperation());
  getOperation().walk([](Operation* op) {
    for (Region& region : op->getRegions()) {
      for (Block& block : region.getBlocks()) {
        if (block.empty()) continue;
        auto ops = SortBlockTopologically(block, OutsideCompilationOrdering);
        // Replace the block with the reordered block.
        for (Operation* o : ops) {
          o->remove();
          block.push_back(o);
        }
      }
    }
  });
  UnmarkCompilationOps(getOperation());
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateMoveTpuCompileToFrontPass() {
  return std::make_unique<MoveTpuCompileToFrontPass>();
}

}  // namespace TF
}  // namespace mlir
