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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/op_utils.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORUNDOMERGECONSTACROSSMESH
#define GEN_PASS_DEF_DTENSORELIDEIDENTITYBEFORECOPYTOMESH
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

// MLIR pass that undoes unintended const merging across different meshes within
// the same Block by canonicalization passes.
struct DTensorUndoMergeConstAcrossMesh
    : public impl::DTensorUndoMergeConstAcrossMeshBase<
          DTensorUndoMergeConstAcrossMesh> {
  void runOnOperation() override { DuplicateConstants(getOperation()); }
};

struct DTensorElideIdentityBeforeCopyToMesh
    : public impl::DTensorElideIdentityBeforeCopyToMeshBase<
          DTensorElideIdentityBeforeCopyToMesh> {
  void runOnOperation() override {
    getOperation().walk([](mlir::TF::CopyToMeshGradOp op) {
      mlir::Value input_value = op->getOperand(0);
      mlir::Operation* defining_op = input_value.getDefiningOp();
      if (!mlir::isa<mlir::TF::IdentityOp>(defining_op)) {
        return;
      }
      op->replaceUsesOfWith(input_value, defining_op->getOperand(0));
      if (!defining_op->use_empty()) {
        return;
      }
      defining_op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorUndoMergeConstAcrossMesh() {
  return std::make_unique<DTensorUndoMergeConstAcrossMesh>();
}
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorElideIdentityBeforeCopyToMesh() {
  return std::make_unique<DTensorElideIdentityBeforeCopyToMesh>();
}
}  // namespace dtensor
}  // namespace machina
