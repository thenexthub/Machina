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

#include <memory>
#include <vector>

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_REPLICATETENSORLISTINITOPSPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// Replicates the TensorList initialization ops for all the uses.
// No need to delete the original TensorList as it might be used elsewhere.
template <typename T>
void ReplicateTensorListForUses(T tensor_list_op) {
  Value tensor_list = tensor_list_op.getResult();
  std::vector<OpOperand*> uses;
  for (auto& use : tensor_list.getUses()) {
    uses.emplace_back(&use);
  }
  OpBuilder builder(tensor_list_op.getOperation());
  for (OpOperand* operand : uses) {
    auto new_op = builder.clone(*tensor_list_op.getOperation());
    operand->set(new_op->getResult(0));
  }
}

// This transformation pass replicates TensorList initialization ops.
class ReplicateTensorListInitOps
    : public impl::ReplicateTensorListInitOpsPassBase<
          ReplicateTensorListInitOps> {
 public:
  void runOnOperation() override {
    getOperation().walk([](Operation* op) {
      if (auto tl_reserve = dyn_cast<TensorListReserveOp>(op)) {
        ReplicateTensorListForUses(tl_reserve);
      }
      if (auto tl_empty = dyn_cast<EmptyTensorListOp>(op)) {
        ReplicateTensorListForUses(tl_empty);
      }
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicateTensorListInitOpsPass() {
  return std::make_unique<ReplicateTensorListInitOps>();
}

}  // namespace TF
}  // namespace mlir
