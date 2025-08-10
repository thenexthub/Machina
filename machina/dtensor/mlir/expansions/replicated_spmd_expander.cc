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

#include "machina/dtensor/mlir/expansions/replicated_spmd_expander.h"

#include <vector>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/FormatVariadic.h"
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
#include "machina/dtensor/mlir/op_utils.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/spmd_expander_common.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

// Relayout all operands and outputs to replicated layout.
StatusOr<mlir::Operation*>
ReplicatedOpSPMDExpander::ReplicatedRelayoutOperandsAndOutputs(
    mlir::Operation* op, const std::vector<Layout>& operand_layouts,
    const std::vector<Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Relayout operands
  for (auto i = 0; i < operand_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
    TF_ASSIGN_OR_RETURN(
        const auto new_operand,
        EmitRelayout(op->getOperand(i), operand_layouts[i], new_layout));
    op->setOperand(i, new_operand);
  }
  // Expand to local shape
  op = InferSPMDExpandedLocalShape(op);

  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  toolchain::SmallVector<mlir::Value, 4> generated_outputs;
  toolchain::SmallVector<mlir::Type, 4> generated_types;

  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;

  // Relayout outputs
  for (auto i = 0; i < output_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
    TF_ASSIGN_OR_RETURN(auto new_output,
                        EmitRelayout(op->getOpResult(i), new_layout,
                                     output_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(new_output);
    generated_types.emplace_back(new_output.getType());
    if (last_op_after_splitting->isBeforeInBlock(new_output.getDefiningOp())) {
      last_op_after_splitting = new_output.getDefiningOp();
    }
  }
  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(last_op_after_splitting);

  // Tie all outputs together with identity_n
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); ++i) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }

  return identity_op.getOperation();
}

StatusOr<mlir::Operation*> ReplicatedOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  if (relayout_when_sharded_)
    return ReplicatedRelayoutOperandsAndOutputs(op, operand_layouts,
                                                output_layouts);
  if (!AllReplicated(output_layouts) || !AllReplicated(operand_layouts)) {
    return errors::InvalidArgument(
        toolchain::formatv("Expecting {0} to have input and output layouts to be "
                      "fully replicated but was not. ",
                      OpName(op))
            .str());
  }
  return op;
}

// Always return a set of replicated layouts.
StatusOr<toolchain::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

// Always return a set of replicated layouts.
StatusOr<toolchain::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace machina
