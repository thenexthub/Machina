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

#include "machina/dtensor/mlir/expansions/control_flow_spmd_expander.h"

#include <cassert>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> WhileRegionSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  assert(op->getNumOperands() == op->getNumResults());
  // Set the type for the results of the WhileRegion explicitly.
  //
  // Normally we would use InferSPMDExpandedLocalShape for this, but that
  // function requires the op to either have a type inference interface
  // (which WhileRegion does not) or a TensorFlow ShapeFn (WhileRegion is not
  // a TensorFlow op). During the standard MLIR shape inference pass this op
  // is handled by a special case in InferShapeForSingleOperation.
  for (int i = 0; i < op->getNumOperands(); ++i)
    op->getResult(i).setType(op->getOperand(i).getType());

  auto while_op = toolchain::cast<mlir::TF::WhileRegionOp>(op);
  for (const auto& data :
       toolchain::enumerate(toolchain::zip(while_op.getCond().front().getArguments(),
                                 while_op.getBody().front().getArguments()))) {
    const int index = data.index();
    mlir::BlockArgument cond_arg = std::get<0>(data.value());
    mlir::BlockArgument body_arg = std::get<1>(data.value());
    cond_arg.setType(while_op.getOperand(index).getType());
    body_arg.setType(while_op.getOperand(index).getType());
  }

  return op;
}

StatusOr<toolchain::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<toolchain::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<mlir::Operation*> IfRegionSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto if_op = toolchain::cast<mlir::TF::IfRegionOp>(op);
  for (mlir::Value result : if_op->getResults()) {
    auto result_layout_op = toolchain::dyn_cast_or_null<mlir::TF::DTensorLayout>(
        *result.getUsers().begin());
    if (!result_layout_op)
      return errors::InvalidArgument(
          "Missing layout of If op result during SPMD expansion.");

    const Layout layout = result_layout_op.getLayout();
    if (!layout.IsFullyReplicated()) {
      const auto global_shape = result_layout_op.getGlobalShape();
      if (!global_shape)
        return errors::InvalidArgument(
            "Shape of If op must be statically known for SPMD expansion.");

      result.setType(mlir::RankedTensorType::get(
          layout.LocalShapeFromGlobalShape(*global_shape),
          mlir::cast<mlir::TensorType>(result.getType()).getElementType()));
    }
  }
  return op;
}

StatusOr<toolchain::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // No-op for forward propagation.
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<toolchain::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // Layout propagation for TF::IfRegion op is no-op. Actual layout
  // propagation logic depends on layout propgation of ops inside the
  // then/else regions of the IfRegion op.
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)))}});
}

}  // namespace dtensor
}  // namespace machina
