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

#include "machina/dtensor/mlir/expansions/bias_add_spmd_expander.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallPtrSet.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops_a_m.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

namespace {

int get_c_dimension_idx(const Layout& layout, toolchain::StringRef data_format) {
  // If format is "N...C", the bias is added to the last dimension.
  int c_dim_idx = layout.sharding_spec_strs().size() - 1;
  if (data_format.starts_with("NC")) {
    // If format is "NC...", the bias is added to the 'C' dimension.
    c_dim_idx = layout.sharding_spec_strs().size() - 3;
  }
  return c_dim_idx;
}

}  // namespace

StatusOr<mlir::Operation*> BiasAddExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  mlir::TF::BiasAddOp bias_add_op = toolchain::cast<mlir::TF::BiasAddOp>(op);
  const toolchain::StringRef data_format = bias_add_op.getDataFormat();
  const int c_dim_idx = get_c_dimension_idx(output_layout, data_format);

  // Bias add op has 2 inputs: value and bias.
  assert(op->getOpOperands().size() == 2);
  mlir::OpOperand& input = op->getOpOperand(0);
  TF_ASSIGN_OR_RETURN(Layout input_layout,
                      ExtractRequiredLayoutFromOperand(input.get()));

  mlir::OpOperand& bias = op->getOpOperand(1);

  TF_ASSIGN_OR_RETURN(const Layout bias_layout,
                      ExtractRequiredLayoutFromOperand(bias.get()));

  // Check if output is sharded more, change input layout to match output
  // layout.
  int64_t num_input_shards = input_layout.num_shards_for_dim(c_dim_idx);
  int64_t num_output_shards = output_layout.num_shards_for_dim(c_dim_idx);

  if (num_input_shards < num_output_shards) {
    mlir::Value output;
    std::vector<std::string> input_new_specs =
        output_layout.sharding_spec_strs();
    TF_ASSIGN_OR_RETURN(
        const Layout new_input_layout,
        Layout::GetLayout(input_new_specs, input_layout.mesh()));
    TF_ASSIGN_OR_RETURN(
        output, EmitRelayout(input.get(), input_layout, new_input_layout));
    input.set(output);
    input_layout = new_input_layout;
  }

  // Map bias layout sharding to match sharding for 'c' dimension of input, if
  // not same already.
  if (bias_layout.sharding_spec(0) != input_layout.sharding_spec(c_dim_idx)) {
    mlir::Value output;

    std::vector<std::string> bias_new_specs = {
        input_layout.sharding_spec_strs()[c_dim_idx]};
    TF_ASSIGN_OR_RETURN(const Layout new_bias_layout,
                        Layout::GetLayout(bias_new_specs, bias_layout.mesh()));
    TF_ASSIGN_OR_RETURN(output,
                        EmitRelayout(bias.get(), bias_layout, new_bias_layout));
    bias.set(output);
  }

  // Perform SPMD operation locally
  mlir::Operation* new_local_op = InferSPMDExpandedLocalShape(op);

  // Convert result layout to output layout.
  toolchain::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  TF_ASSIGN_OR_RETURN(mlir::Value relayout_output,
                      EmitRelayout(new_local_op->getOpResult(0), input_layout,
                                   output_layout, &newly_created_ops));
  op->getResult(0).replaceAllUsesExcept(relayout_output, newly_created_ops);
  return relayout_output.getDefiningOp();
}

StatusOr<toolchain::DenseMap<int, Layout>> BiasAddExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // If we do not have an input layout then do not infer an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  Layout input_layout = input_layouts.lookup(0);
  mlir::TF::BiasAddOp bias_add_op = toolchain::cast<mlir::TF::BiasAddOp>(op);
  toolchain::StringRef data_format = bias_add_op.getDataFormat();
  int c_dim_idx = get_c_dimension_idx(input_layout, data_format);

  std::vector<std::string> new_output_layout_specs =
      input_layout.sharding_spec_strs();
  if (Layout::IsUnshardedDimension(new_output_layout_specs[c_dim_idx]) &&
      input_layouts.find(1) != input_layouts.end()) {
    // Shard c_dim using bias sharding as long as the sharding spec is not
    // already used in input for some other dimension.
    Layout bias_layout = input_layouts.lookup(1);
    std::string bias_sharding = bias_layout.sharding_spec(0);
    if (std::find(new_output_layout_specs.begin(),
                  new_output_layout_specs.end(),
                  bias_sharding) == new_output_layout_specs.end()) {
      new_output_layout_specs[c_dim_idx] = bias_layout.sharding_spec(0);
    }
  }
  TF_ASSIGN_OR_RETURN(
      Layout new_output_layout,
      Layout::GetLayout(new_output_layout_specs, input_layout.mesh()));

  return toolchain::DenseMap<int, Layout>({{0, new_output_layout}});
}

StatusOr<toolchain::DenseMap<int, Layout>> BiasAddExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  toolchain::DenseMap<int, Layout> input_layouts;

  // If output layout is given, match input_layout and bias layout to match
  // it.
  Layout output_layout = output_layouts.lookup(0);

  // Bias layout should match 'C' dimension of input layout.
  mlir::TF::BiasAddOp bias_add_op = toolchain::cast<mlir::TF::BiasAddOp>(op);
  toolchain::StringRef data_format = bias_add_op.getDataFormat();
  const int c_dim_idx = get_c_dimension_idx(output_layout, data_format);

  std::vector<std::string> bias_new_specs = {
      output_layout.sharding_spec_strs()[c_dim_idx]};
  TF_ASSIGN_OR_RETURN(Layout new_bias_layout,
                      Layout::GetLayout(bias_new_specs, output_layout.mesh()));

  return toolchain::DenseMap<int, Layout>(
      {{0, output_layout}, {1, new_bias_layout}});
}
}  // namespace dtensor
}  // namespace machina
