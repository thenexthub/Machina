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

#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUCOLOCATESPLITSPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

constexpr char kDeviceAttr[] = "device";
// Attribute of colocation classes.
constexpr char kClassAttr[] = "_class";

bool HasDevice(Operation* op) {
  auto attr = op->getAttrOfType<StringAttr>(kDeviceAttr);
  if (!attr) return false;
  return !attr.getValue().empty();
}

// Returns the predecessors of `op` when `op`'s predecessors are wrapped by
// islands.
toolchain::SmallVector<Operation*> IslandPredecessors(Operation* op) {
  toolchain::SmallVector<Operation*> predecessors;
  for (Value operand : op->getOperands()) {
    if (Operation* pred = operand.getDefiningOp()) {
      int result_number = toolchain::cast<OpResult>(operand).getResultNumber();
      if (auto pred_island = toolchain::dyn_cast<tf_executor::IslandOp>(pred)) {
        Value yield_operand = pred_island.GetYield().getOperand(result_number);
        predecessors.push_back(yield_operand.getDefiningOp());
      }
    }
  }
  return predecessors;
}

struct TPUColocateSplits
    : public impl::TPUColocateSplitsPassBase<TPUColocateSplits> {
  void runOnOperation() override;
};

void TPUColocateSplits::runOnOperation() {
  getOperation().walk([&](Operation* op) {
    if (auto split = toolchain::dyn_cast<TF::SplitOp>(op)) {
      if (HasDevice(split) || split->getAttrOfType<ArrayAttr>(kClassAttr))
        return WalkResult::advance();
      for (Operation* pred : IslandPredecessors(split)) {
        if (auto colocation_classes =
                pred->getAttrOfType<ArrayAttr>(kClassAttr)) {
          split->setAttr(kClassAttr, colocation_classes);
          return WalkResult::advance();
        }
      }
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTPUColocateSplitsPass() {
  return std::make_unique<TPUColocateSplits>();
}

}  // namespace TFTPU
}  // namespace mlir
