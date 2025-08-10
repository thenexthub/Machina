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

#include "machina/dtensor/mlir/expansions/fill_spmd_expander.h"

#include <optional>

#include "toolchain/Support/Casting.h"
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/xla/mlir_hlo/utils/convert_op_folder.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/types.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_location.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/spmd_expander_common.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> FillSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto original_fill = mlir::cast<mlir::TF::FillOp>(op);
  TF_ASSIGN_OR_RETURN(auto dims_layout,
                      ExtractLayoutFromOperand(original_fill.getDims()));
  if (!dims_layout.has_value()) {
    return errors::InvalidArgument(
        "Failed during SPMD expansion of tf.FillOp. Layout of dimension "
        "input must be known.");
  }

  if (!dims_layout->IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Expected the layout for fill's `dims` argument to be fully "
        "replicated. Got ",
        dims_layout->ToString());
  }
  TF_ASSIGN_OR_RETURN(std::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  if (!output_layout.has_value())
    return errors::Internal(
        "FillOp doesn't have a layout after layout propagation");
  if (output_layout->IsFullyReplicated()) {
    // For fully replicated layouts the local shape on each device is the same
    // as the global shape.
    return InferSPMDExpandedLocalShape(op);
  }

  // For sharded outputs, the `dims` just needs to be translated from the
  // global to the local shape.
  mlir::OpBuilder builder(op->getBlock(), ++mlir::Block::iterator(op));

  // Create a tensor from the sharding spec, with the dtype of the original
  // attribute.
  auto shard_values = output_layout->num_shards();
  auto int_type = mlir::RankedTensorType::get(
      static_cast<int64>(shard_values.size()), builder.getIntegerType(32));
  auto int_attr = mlir::DenseIntElementsAttr::get(int_type, shard_values);
  auto target_type_attr = mlir::hlo::convertElementsAttr(
      int_attr, mlir::cast<mlir::TensorType>(original_fill.getDims().getType())
                    .getElementType());

  auto location = DT_LOC(op);
  auto num_shards =
      mlir::TF::ConstOp::create(builder, location, target_type_attr);
  // Divide the global shape by the sharding spec.
  auto div = mlir::TF::DivOp::create(builder, location, original_fill.getDims(),
                                     num_shards.getResult());
  // Copy over static shape information if available
  auto global_output_type =
      mlir::cast<mlir::TensorType>(original_fill.getResult().getType());
  TF_ASSIGN_OR_RETURN(
      auto local_type,
      LocalTypeFromGlobalType(output_layout.value(), global_output_type));

  auto new_fill = mlir::TF::FillOp::create(
      builder, location, local_type, div.getResult(), original_fill.getValue());
  original_fill.getResult().replaceAllUsesWith(new_fill.getOutput());
  original_fill.erase();
  return InferSPMDExpandedLocalShape(new_fill);
}

StatusOr<toolchain::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for output.
  return toolchain::DenseMap<int, Layout>(
      {{0,
        Layout::ReplicatedOnMesh(
            mesh, ValueRank(toolchain::cast<mlir::TF::FillOp>(op).getOutput()))}});
}

StatusOr<toolchain::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for dims / value operand of Fill op.
  return toolchain::DenseMap<int, Layout>({{0, Layout::ReplicatedOnMesh(mesh, 1)},
                                      {1, Layout::ReplicatedOnMesh(mesh, 0)}});
}

}  // namespace dtensor
}  // namespace machina
