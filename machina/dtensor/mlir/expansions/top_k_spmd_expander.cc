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

#include "machina/dtensor/mlir/expansions/top_k_spmd_expander.h"

#include <string>
#include <vector>

#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/IRMapping.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/proto/layout.pb.h"

namespace machina {
namespace dtensor {

// layout -> layout[:-1] + unsharded
StatusOr<Layout> GetSuggestedLayout(const Layout& input_layout) {
  std::vector<std::string> layout_specs(input_layout.rank());

  for (int i = 0; i < input_layout.rank() - 1; ++i) {
    layout_specs[i] = input_layout.sharding_spec(i);
  }
  layout_specs[input_layout.rank() - 1] = Layout::kUnshardedDim;
  return Layout::GetLayout(layout_specs, input_layout.mesh());
}

StatusOr<mlir::Operation*> TopKSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto top_k_op = mlir::cast<mlir::TF::TopKV2Op>(op);
  mlir::Value input = top_k_op.getInput();
  TF_ASSIGN_OR_RETURN(auto input_layout, ExtractLayoutFromOperand(input));

  if (!input_layout)
    return errors::InvalidArgument(
        "layout of TopKV2Op must be known before SPMD expansion.");

  TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op));
  for (const auto& layout : layouts) {
    if (layout.has_value() && !layout->IsLastDimReplicated()) {
      return errors::InvalidArgument(
          "The last dimensions of TopKV2Op outputs should be UNSHARDED.");
    }
  }
  mlir::OpBuilder builder(op);
  if (!input_layout->IsLastDimReplicated()) {
    TF_ASSIGN_OR_RETURN(Layout new_layout, GetSuggestedLayout(*input_layout));
    TF_ASSIGN_OR_RETURN(
        input, EmitAllGather(builder, input, *input_layout, new_layout));
    mlir::IRMapping mapping;
    mapping.map(op->getOperand(0), input);
    mlir::Operation* new_op = builder.clone(*op, mapping);
    new_op = InferSPMDExpandedLocalShape(new_op);

    op->getResult(0).replaceAllUsesWith(new_op->getResult(0));
    op->getResult(1).replaceAllUsesWith(new_op->getResult(1));
    op->erase();
    return new_op;
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<toolchain::DenseMap<int, Layout>> TopKSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      GetSuggestedLayout(input_layouts.lookup(0)));

  return toolchain::DenseMap<int, Layout>({
      {0, output_layout},  // values
      {1, output_layout},  // indices
  });
}

StatusOr<toolchain::DenseMap<int, Layout>> TopKSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // If the output values layout is missing, don't return an input layout.
  if (output_layouts.find(0) == output_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(Layout input_layout,
                      GetSuggestedLayout(output_layouts.lookup(0)));
  const Mesh& mesh = input_layout.mesh();

  return toolchain::DenseMap<int, Layout>({
      {0, input_layout},                                // input
      {1, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)},  // k
  });
}

}  // namespace dtensor
}  // namespace machina
