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

#include "machina/dtensor/mlir/expansions/qr_spmd_expander.h"

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/proto/layout.pb.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> QRSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));

  // Relayout all layouts to the first output layout with the last two
  // dimensions replicated. We can do more optimization but this is fine
  TF_ASSIGN_OR_RETURN(
      Layout new_layout,
      output_layouts[0].GetLayoutWithReducedDims({-1, -2}, /*keep_dims=*/true));

  TF_ASSIGN_OR_RETURN(
      const auto new_operand,
      EmitRelayout(op->getOperand(0), operand_layouts[0], new_layout));
  op->setOperand(0, new_operand);

  mlir::OpBuilder builder(op);
  op = InferSPMDExpandedLocalShape(op);

  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  toolchain::SmallVector<mlir::Value, 4> generated_outputs;
  toolchain::SmallVector<mlir::Type, 4> generated_types;

  // Relayout outputs
  for (auto i = 0; i < output_layouts.size(); i++) {
    TF_ASSIGN_OR_RETURN(auto new_output,
                        EmitRelayout(op->getOpResult(i), new_layout,
                                     output_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(new_output);
    generated_types.emplace_back(new_output.getType());
  }
  if (generated_outputs[0].getDefiningOp()->isBeforeInBlock(
          generated_outputs[1].getDefiningOp()))
    builder.setInsertionPointAfterValue(generated_outputs[1]);
  else
    builder.setInsertionPointAfterValue(generated_outputs[0]);

  // Tie the two outputs together with an identity op
  auto identity_op = mlir::TF::IdentityNOp::create(
      builder, op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); i++) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }
  return identity_op.getOperation();
}

StatusOr<toolchain::DenseMap<int, Layout>> QRSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // If no input layout at index 0 is present then do not infer any output
  // layout.
  if (input_layouts.find(0) == input_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  // Set the output layouts as the copy of the input layouts with the last 2
  // dimensions replicated.
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      input_layouts.lookup(0).GetLayoutWithReducedDims(
                          {-1, -2}, /*keep_dims=*/true));
  return toolchain::DenseMap<int, Layout>({{0, output_layout}, {1, output_layout}});
}

StatusOr<toolchain::DenseMap<int, Layout>> QRSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // If no output layout is present then do not infer any operand layouts.
  if (output_layouts.empty()) return toolchain::DenseMap<int, Layout>();

  // Set the operand layout as the copy of the output layouts with the last 2
  // dimensions replicated.
  Layout layout = output_layouts.find(0) != output_layouts.end()
                      ? output_layouts.lookup(0)
                      : output_layouts.lookup(1);
  TF_ASSIGN_OR_RETURN(
      Layout layout_reduced_dims,
      layout.GetLayoutWithReducedDims({-1, -2}, /*keep_dims=*/true));
  return toolchain::DenseMap<int, Layout>({{0, layout_reduced_dims}});
}

}  // namespace dtensor
}  // namespace machina
