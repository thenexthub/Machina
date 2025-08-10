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

#include "machina/dtensor/mlir/expansions/nullary_spmd_expander.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> NullarySPMDExpander::ExpandOp(mlir::Operation* op) {
  if (op->getNumResults() == 0) return op;

  bool all_operands_fully_replicated = true;
  TF_ASSIGN_OR_RETURN(auto op_layouts, ExtractLayoutFromOp(op));
  for (const auto& op_layout : op_layouts) {
    if (!op_layout)
      return errors::InvalidArgument(
          "Nullary op layouts must be known before SPMD expansion.");
    all_operands_fully_replicated =
        all_operands_fully_replicated && op_layout->IsFullyReplicated();
  }

  if (all_operands_fully_replicated) return op;

  if (auto const_op = mlir::dyn_cast<mlir::TF::ConstOp>(op)) {
    if (auto dense =
            mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue())) {
      if (dense.isSplat()) {
        // A 'splat' value for a DenseElementsAttr, has a single value for
        // all its elements. For these inputs, we don't need to slice. We just
        // need to update the shape of the attribute given the requested
        // sharding.
        assert(dense.getType().getRank() == op_layouts[0]->rank());
        auto shape = dense.getType().getShape();
        std::vector<int64_t> new_shape(dense.getType().getRank());
        for (int i = 0; i < op_layouts[0]->rank(); ++i) {
          const int num_shards = op_layouts[0]->num_shards_for_dim(i);
          if (shape[i] % num_shards != 0)
            return errors::InvalidArgument(
                "has output dimension size ", shape[i],
                " which is not evenly divisible by the number of shards ",
                num_shards, " in the layout for that dimension.");
          new_shape[i] = shape[i] / num_shards;
        }
        const_op.setValueAttr(mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(new_shape,
                                        dense.getType().getElementType()),
            dense.getSplatValue<mlir::Attribute>()));
        return InferSPMDExpandedLocalShape(op);
      }
    }
  }

  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  toolchain::SmallVector<mlir::Value, 4> generated_outputs;
  toolchain::SmallVector<mlir::Type, 4> generated_types;

  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (int i = 0; i < op_layouts.size(); ++i) {
    // Split each output to the correct layout by assuming the input is
    // replicated.
    TF_ASSIGN_OR_RETURN(
        const mlir::Value output,
        EmitAllScatter(builder, op->getOpResult(i),
                       Layout::ReplicatedOnMesh(op_layouts[i]->mesh(),
                                                op_layouts[i]->rank()),
                       *op_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(output);
    generated_types.emplace_back(output.getType());
  }

  auto identity_op = mlir::TF::IdentityNOp::create(
      builder, op->getLoc(), generated_types, generated_outputs);

  newly_created_ops.insert(identity_op);
  for (int i = 0; i < op_layouts.size(); ++i)
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);

  return identity_op.getOperation();
}

StatusOr<toolchain::DenseMap<int, Layout>> NullarySPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  auto enclosing_mesh = op->getParentOfType<mlir::tf_device::ClusterOp>();
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshFromOp(enclosing_mesh));
  if (!mesh.has_value())
    return errors::Internal("Failure in extracting mesh from Nullary Op.");
  toolchain::DenseMap<int, Layout> output_layouts;
  // Nullary ops always output replicated layout for output values.
  for (auto i = 0; i < op->getNumResults(); ++i) {
    auto output_ranked_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(i).getType());
    if (!output_ranked_type) {
      return errors::InvalidArgument(
          toolchain::formatv("requires output type to have statically known rank, "
                        "but got : {0}",
                        output_ranked_type)
              .str());
    }
    output_layouts[i] =
        Layout::ReplicatedOnMesh(*mesh, output_ranked_type.getRank());
  }
  return output_layouts;
}

StatusOr<toolchain::DenseMap<int, Layout>>
NullarySPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // No operand inputs.
  return toolchain::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace machina
