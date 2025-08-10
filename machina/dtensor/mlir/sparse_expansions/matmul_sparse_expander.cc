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

#include "machina/dtensor/mlir/sparse_expansions/matmul_sparse_expander.h"

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/mlir/sparse_expander_common.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> MatMulSparseExpander::ExpandOp(mlir::Operation* op) {
  mlir::TF::MatMulOp mm = mlir::cast<mlir::TF::MatMulOp>(op);
  // If any of the transpose attributes are true, then return original op.
  if (mm.getTransposeA() || mm.getTransposeB()) return op;

  // Expand to SparseTensorDenseMatMul Op only if the left operand
  // is a SparseTensor.
  if (IsSparseValue(op->getOperand(0)) && !IsSparseValue(op->getOperand(1))) {
    mlir::OpBuilder builder(op);
    // Since operand 0 is a SparseValue, we don't need to check that
    // the indices, values, and dense_shapes exist.
    mlir::TF::SparseTensorDenseMatMulOp new_op =
        builder.create<mlir::TF::SparseTensorDenseMatMulOp>(
            op->getLoc(), op->getResultTypes(),
            mlir::ValueRange{
                GetIndicesFromSparseTensor(op->getOperand(0)).value(),
                GetValuesFromSparseTensor(op->getOperand(0)).value(),
                GetDenseShapesFromSparseTensor(op->getOperand(0)).value(),
                op->getOperand(1)});

    op->getResult(0).replaceAllUsesWith(new_op.getResult());
    op->erase();
    return new_op.getOperation();
  }

  // Any other case, return the original op.
  return op;
}

}  // namespace dtensor
}  // namespace machina
