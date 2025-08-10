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

#include "machina/dtensor/mlir/expansions/tensorlist_setitem_spmd_expander.h"

#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> TensorListSetItemSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  TF_ASSIGN_OR_RETURN(const auto output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  // If the layout of handle and layout of output handle is not equal,
  // then return an error. This is probably not possible to have different
  // handles have differing layouts since this means we are changing the shapes
  // and have to relayout all the items that this points to.
  if (operand_layouts[0] != output_layout) {
    return errors::Internal(
        "Differing layouts for variant tensor input and variant tensor output "
        "is not yet allowed.");
  }

  // Relayout 'item' tensor to the layout of the 'handle',
  TF_ASSIGN_OR_RETURN(
      const auto new_item,
      EmitRelayout(op->getOperand(2), operand_layouts[2], operand_layouts[0]));

  op->setOperand(2, new_item);
  // We do not need to relayout the output since above we checked that
  // the input list handle and output list handle is the same.
  return op;
}

StatusOr<toolchain::DenseMap<int, Layout>>
TensorListSetItemSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(auto global_shape, GetShapeOfValue(op->getOpResult(0)));
  // Prefer the output list handle layout to be the same layout
  // as the input list handle layout.
  if (input_layouts.find(0) != input_layouts.end())
    return toolchain::DenseMap<int, Layout>({{0, input_layouts.lookup(0)}});
  // Assume replicated layout if the list handle layout does not exist.
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, global_shape.size())}});
}

StatusOr<toolchain::DenseMap<int, Layout>>
TensorListSetItemSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // Prefer the layout of `item` tensor to be the layout of the output list
  // handle.
  if (output_layouts.find(0) != output_layouts.end()) {
    return toolchain::DenseMap<int, Layout>({{2, output_layouts.lookup(0)}});
  }
  return toolchain::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace machina
