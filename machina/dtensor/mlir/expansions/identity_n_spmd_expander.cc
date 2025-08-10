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

#include "machina/dtensor/mlir/expansions/identity_n_spmd_expander.h"

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> IdentityNSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op));

  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  toolchain::SmallVector<mlir::Value, 4> generated_outputs;
  toolchain::SmallVector<mlir::Type, 4> generated_types;

  mlir::OpBuilder builder(op);
  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;
  for (int i = 0; i < layouts.size(); ++i) {
    auto output_layout = layouts[i];
    if (!output_layout)
      return errors::InvalidArgument(
          "layout of (", i,
          "-th output of IdentityNOp must be known before SPMD expansion.");

    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractLayoutFromOperand(op->getOperand(i)));
    if (!operand_layout)
      return errors::InvalidArgument(
          "layout of (", i,
          "-th input of IdentityNOp must be known before SPMD expansion.");

    TF_ASSIGN_OR_RETURN(const mlir::Value output,
                        EmitRelayout(op->getOperand(i), *operand_layout,
                                     *output_layout, &newly_created_ops));
    generated_outputs.emplace_back(output);
    generated_types.emplace_back(output.getType());
    // InsertionPoint has to come after all newly created Ops.
    if (last_op_after_splitting->isBeforeInBlock(output.getDefiningOp())) {
      last_op_after_splitting = output.getDefiningOp();
    }
  }

  builder.setInsertionPointAfter(last_op_after_splitting);
  auto identity_op = mlir::TF::IdentityNOp::create(
      builder, op->getLoc(), generated_types, generated_outputs);

  for (int i = 0; i < layouts.size(); ++i)
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);

  op->erase();

  return InferSPMDExpandedLocalShape(identity_op.getOperation());
}

StatusOr<toolchain::DenseMap<int, Layout>>
IdentityNSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  return input_layouts;
}

StatusOr<toolchain::DenseMap<int, Layout>>
IdentityNSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  return output_layouts;
}

}  // namespace dtensor
}  // namespace machina
