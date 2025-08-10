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

#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/sparse_expander.h"
#include "machina/dtensor/mlir/topological_iterator.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORSPARSEEXPANSION
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kMainFunctionName[] = "main";

// Expand every op that consumes SparseTensor operands in topological order.
mlir::LogicalResult ConductSparseExpansion(mlir::ModuleOp module) {
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>(kMainFunctionName);
  if (!main_func)
    return module.emitOpError(
        "could not find `main` function in module for SPMD expansion.");

  TopologicalIterator iterator(main_func);
  while (iterator.hasNext()) {
    mlir::Operation* op = iterator.next();

    mlir::Operation* expanded_op = nullptr;
    auto status = RunSparseExpansion(op, &expanded_op);
    if (!status.ok() || expanded_op == nullptr) {
      // Sometimes op may been erased and expanded_op set.
      // In this case we should emit the error on the expanded op.
      mlir::Operation* emit_op = op;
      if (expanded_op != nullptr) emit_op = expanded_op;
      return emit_op->emitError(WithContext(status, __FILE__, __LINE__,
                                            "While computing Sparse expansion")
                                    .message());
    }
  }
  return mlir::success();
}

// After Sparse Expansion pass, there may be unused SparseToDenseOps due to
// expanded ops possibly taking the operands of the SparseToDenseOps instead
// of the output of the SparseToDenseOps. So remove unused SparseToDenseOps
// and its corresponding dependent ops like DTensorLayout and Const ops.
void RemoveUnusedSparseToDenseOps(mlir::ModuleOp module) {
  toolchain::SmallVector<mlir::TF::SparseToDenseOp, 4> sparse_ops_to_erase;
  toolchain::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops_to_erase;

  module.walk([&](mlir::TF::SparseToDenseOp op) {
    // Delete this op if it either has no consuming ops or the only consuming
    // op is a DTensorLayout op that also has no consuming ops.
    if (op->use_empty()) {
      sparse_ops_to_erase.emplace_back(op);
    } else if (op->hasOneUse()) {
      if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(
              op->getOpResult(0).getUses().begin().getUser())) {
        if (layout_op.use_empty()) {
          layout_ops_to_erase.emplace_back(layout_op);
          sparse_ops_to_erase.emplace_back(op);
        }
      }
    }
  });

  // First delete Layout ops and then delete SparseToDense ops.
  for (auto op : layout_ops_to_erase) op.erase();
  for (auto op : sparse_ops_to_erase) {
    // Also delete the corresponding Const ops that are no longer used
    // attached to the SparseToDense ops.
    auto const_op = op.getOperand(3).getDefiningOp();
    op.erase();
    if (const_op->use_empty()) const_op->erase();
  }
}

struct DTensorSparseExpansion
    : public impl::DTensorSparseExpansionBase<DTensorSparseExpansion> {
  void runOnOperation() override {
    auto module = getOperation();
    if (failed(ConductSparseExpansion(module))) return signalPassFailure();

    // After Sparse Expansion, we may no longer use any SparseToDenseOp outputs,
    // so remove them if they are not used.
    RemoveUnusedSparseToDenseOps(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseExpansion() {
  return std::make_unique<DTensorSparseExpansion>();
}

}  // namespace dtensor
}  // namespace machina
