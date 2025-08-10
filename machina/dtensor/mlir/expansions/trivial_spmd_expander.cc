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

#include "machina/dtensor/mlir/expansions/trivial_spmd_expander.h"

#include <cassert>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> TerminatorSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto terminator_op = toolchain::cast<mlir::tf_device::ReturnOp>(op);
  auto parent_op = op->getParentOp();
  auto output_types = toolchain::to_vector<8>(terminator_op.getOperandTypes());
  assert(output_types.size() == parent_op->getNumResults());

  for (const auto& output_type_and_index : toolchain::enumerate(output_types)) {
    const int index = output_type_and_index.index();
    const auto& type = output_type_and_index.value();
    parent_op->getResult(index).setType(type);
  }
  return op;
}

StatusOr<mlir::Operation*> MetadataSPMDExpander::ExpandOp(mlir::Operation* op) {
  for (auto operand : op->getOperands()) {
    TF_ASSIGN_OR_RETURN(auto input_layout, ExtractLayoutFromOperand(operand));
    if (!input_layout.has_value())
      return errors::Internal(
          "All input layouts to Metadata op must be specified at SPMD "
          "expansion.");

    if (!input_layout->IsFullyReplicated())
      return errors::InvalidArgument(
          "Metadata ops like tf.BroadcastGradientArgs op must have replicated "
          "input layouts.");
  }

  TF_ASSIGN_OR_RETURN(auto result_layouts, ExtractLayoutFromOp(op));
  for (const auto& layout : result_layouts) {
    if (!layout.has_value())
      return errors::Internal(
          "All op result layouts of Metadata op must be specified for SPMD "
          "expansion.");

    if (!layout->IsFullyReplicated()) {
      return errors::InvalidArgument(
          "Metadata ops like tf.BroadcastGradientArgs op must have replicated "
          "output layouts.");
    }
  }
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<toolchain::DenseMap<int, Layout>>
MetadataSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (const auto& result_and_index : toolchain::enumerate(op->getOpResults())) {
    const int index = result_and_index.index();
    auto result = result_and_index.value();
    output_layouts.insert(
        {index, Layout::ReplicatedOnMesh(mesh, ValueRank(result))});
  }
  return output_layouts;
}

StatusOr<toolchain::DenseMap<int, Layout>>
MetadataSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (const auto& operand_and_index : toolchain::enumerate(op->getOperands())) {
    const int index = operand_and_index.index();
    auto operand = operand_and_index.value();
    input_layouts.insert(
        {index, Layout::ReplicatedOnMesh(mesh, ValueRank(operand))});
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace machina
