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

#include <memory>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/FoldUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORCONSTANTFOLDING
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

constexpr int kMaxIteration = 10;

mlir::LogicalResult FoldConstantOp(mlir::OperationFolder& folder,
                                   mlir::TF::ConstOp op) {
  bool changed = false;
  int i = 0;
  // Iterate until convergence or until maxIterations. Deletion of the op as
  // a result of being dead or folded is convergence.
  do {
    changed = false;

    // If the operation is trivially dead - remove it.
    if (isOpTriviallyDead(op)) {
      op->erase();
      return mlir::success();
    }

    // Try to fold this op.
    bool inPlaceUpdate;
    if (succeeded(folder.tryToFold(op, &inPlaceUpdate))) {
      changed = true;
      if (!inPlaceUpdate) {
        return mlir::success();
      }
    }
  } while (changed && ++i < kMaxIteration);
  return mlir::success();
}

// MLIR pass that folds constants that can be removed or deduplicated away.
struct DTensorConstantFolding
    : public impl::DTensorConstantFoldingBase<DTensorConstantFolding> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OperationFolder helper(&context);

    // Collect and fold the operations within the function.
    toolchain::SmallVector<mlir::TF::ConstOp, 8> const_ops;
    getOperation().walk([&](mlir::TF::ConstOp op) { const_ops.push_back(op); });

    // Attempt to fold the specified operation, including handling unused or
    // duplicated constants.
    for (mlir::TF::ConstOp op : toolchain::reverse(const_ops))
      if (mlir::failed(FoldConstantOp(helper, op))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorConstantFolding() {
  return std::make_unique<DTensorConstantFolding>();
}

}  // namespace dtensor
}  // namespace machina
