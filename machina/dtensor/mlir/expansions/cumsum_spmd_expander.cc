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

#include "machina/dtensor/mlir/expansions/cumsum_spmd_expander.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/op_utils.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

namespace {

// Extract `axis` tensor from Cumsum op and return it's positive value, since
// it can be a negative index.
StatusOr<int64_t> GetAxisDimension(mlir::Operation* op) {
  auto cumsum = toolchain::dyn_cast<mlir::TF::CumsumOp>(op);
  if (cumsum == nullptr) {
    return errors::Internal(
        absl::StrCat("Expected Cumsum op but got : ", OpName(op)).c_str());
  }
  TF_ASSIGN_OR_RETURN(int64_t axis_dim,
                      ExtractConstIntFromValue(cumsum.getAxis()));
  int64_t tensor_rank = ValueRank(cumsum.getX());
  // Axis can be in range [-tensor_rank, tensor_rank), so we add tensor_rank
  // to wrap it around.
  if (axis_dim >= -tensor_rank && axis_dim < 0) {
    axis_dim += tensor_rank;
  } else if (axis_dim < -tensor_rank || axis_dim >= tensor_rank) {
    return errors::InvalidArgument(
        "Invalid axis; expected a value in [-tensor_rank, tensor_rank)");
  }
  return axis_dim;
}

}  // namespace

StatusOr<mlir::Operation*> CumsumSPMDExpander::ExpandOp(mlir::Operation* op) {
  StatusOr<int64_t> axis_dim = GetAxisDimension(op);
  if (!axis_dim.ok()) return axis_dim.status();

  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  assert(output_layout);

  // Our intermediate computation layout is the output layout with
  // the axis dimension replicated. So set both the operand and output layout
  // to this intermediate layout.
  TF_ASSIGN_OR_RETURN(Layout intermediate_layout,
                      output_layout->GetLayoutWithReducedDims(
                          {axis_dim.value()}, /*keep_dims=*/true));

  // Relayout operand to intermediate layout.
  mlir::OpBuilder builder(op);
  const auto operand = op->getOperand(0);
  TF_ASSIGN_OR_RETURN(auto operand_layout, ExtractLayoutFromOperand(operand));
  if (!operand_layout)
    return errors::InvalidArgument(
        "input layout of Cumsum op must be known before SPMD "
        "expansion.");

  TF_ASSIGN_OR_RETURN(
      const auto new_operand,
      EmitRelayout(operand, operand_layout.value(), intermediate_layout));
  op->setOperand(0, new_operand);

  op = InferSPMDExpandedLocalShape(op);

  // Relayout output to intermediate layout.
  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(op);
  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(op->getOpResult(0), intermediate_layout,
                                   output_layout.value(), &newly_created_ops));
  op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<toolchain::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (input_layouts.find(0) == input_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  auto input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(
      Layout input_layout_reduced_dims,
      input_layout.GetLayoutWithReducedDims({axis_dim},
                                            /*keep_dims=*/true));
  return toolchain::DenseMap<int, Layout>({{0, input_layout_reduced_dims}});
}

StatusOr<toolchain::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (output_layouts.find(0) == output_layouts.end())
    return toolchain::DenseMap<int, Layout>();
  auto output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(
      Layout output_layout_reduced_dims,
      output_layout.GetLayoutWithReducedDims({axis_dim},
                                             /*keep_dims=*/true));
  return toolchain::DenseMap<int, Layout>({{0, output_layout_reduced_dims}});
}

}  // namespace dtensor
}  // namespace machina
