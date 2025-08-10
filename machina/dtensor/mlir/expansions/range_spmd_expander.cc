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

#include "machina/dtensor/mlir/expansions/range_spmd_expander.h"

#include "absl/status/status.h"
#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> RangeSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout)
    return absl::InvalidArgumentError(
        "layout of RangeOp must be known before SPMD expansion.");

  if (!layout->IsFullyReplicated())
    return absl::InternalError("Shared RangeOp is not supported yet.");

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<toolchain::DenseMap<int, Layout>> RangeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always return a Replicated layout. This will always respect the consumer
  // requested layouts.
  return toolchain::DenseMap<int, Layout>({{0, Layout::ReplicatedOnMesh(mesh, 1)}});
}

StatusOr<toolchain::DenseMap<int, Layout>> RangeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always assign a replicated layout to the operands.
  toolchain::DenseMap<int, Layout> input_layouts;
  for (int i = 0; i < op->getNumOperands(); ++i)
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(i)));
  return input_layouts;
}

}  // namespace dtensor
}  // namespace machina
