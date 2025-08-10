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

#include "machina/dtensor/mlir/expansions/dtensor_op_spmd_expander.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/collectives.h"
#include "machina/dtensor/mlir/dtensor_send_recv.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/op_utils.h"

namespace machina {
namespace dtensor {
namespace {

// FIXME(feyu): This function should take layouts as arguments. It doesn't need
// the ops.
// Validates send/recv layout and mesh configurations. Among other things, this
// checks for below constraints.
// 1. Src/target layouts have non empty mesh.
// 2. Src/target layouts have the same host.
// 3. Src/target layouts are from different mesh.
// 4. One of scr/target layout is from host mesh cluster.
// 5. CPU host cluster mesh has 1 device.
absl::Status ValidateSendRecvLayoutConfiguration(
    mlir::TF::DTensorSend dtensor_send, mlir::TF::DTensorRecv dtensor_recv) {
  // If either one of the send/recv ops has already been lowered, then send/recv
  // configuration has already been verified.
  if (!dtensor_send || !dtensor_recv) return absl::OkStatus();

  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> send_layout_or_null,
                      ExtractLayoutFromOperand(dtensor_send.getInput()));

  if (!send_layout_or_null.has_value())
    return absl::InvalidArgumentError(
        "Input to DTensorSend must have specified layout.");

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(dtensor_recv));

  const Layout& send_layout = send_layout_or_null.value();
  const Layout& recv_layout = output_layout;
  const Mesh& recv_mesh = dtensor_recv.getMesh();

  const Mesh& send_mesh = send_layout.mesh();

  // If any one of send/recv mesh are empty, return error.
  if (send_mesh.IsEmpty() || recv_mesh.IsEmpty())
    return absl::InvalidArgumentError(
        "Found empty mesh when sending/receiving tensor across clusters.");

  // If send host not found in list of receiving hosts, return error.
  std::vector<std::string> send_hosts = send_layout.ReducedMesh().hosts();
  std::vector<std::string> recv_hosts = recv_layout.ReducedMesh().hosts();
  if (send_hosts != recv_hosts)
    return absl::InvalidArgumentError("Send and receive hosts don't match");

  // Check shards in sending host match those in the receiving host.
  const auto send_host_shard_map = send_layout.HostShardMap();
  const auto recv_host_shard_map = recv_layout.HostShardMap();
  for (const std::string& host : send_hosts) {
    const ShardVector& shards_in_send_host =
        send_host_shard_map.find(host)->second;
    ShardVector shards_in_recv_host = recv_host_shard_map.find(host)->second;
    if (shards_in_send_host != shards_in_recv_host)
      return absl::InvalidArgumentError(absl::StrCat(
          "Send and receive host shard vectors don't match. Send shard_vector:",
          shards_in_send_host.ToString(),
          " / Recv host spec : ", shards_in_recv_host.ToString()));
  }

  // Send/Recv mesh must be different.
  if (recv_mesh == send_mesh)
    return absl::InvalidArgumentError(
        "Found CopyToMesh op sending tensor to same mesh. Only use "
        "CopyToMesh to transfer data across different mesh cluster. For "
        "changing layout within the same mesh, use tf.Relayout op.");

  // Either one of send/recv pair must be to/from CPU mesh.
  // For example, TPU mesh -> GPU mesh or TPU mesh -> another TPU mesh
  // is disallowed.
  if (!send_mesh.is_cpu_mesh() && !recv_mesh.is_cpu_mesh())
    return absl::InvalidArgumentError(
        "tf.CopyToMesh op must be used to send data from/to host mesh.");

  return absl::OkStatus();
}

template <typename RelayoutOp>
StatusOr<mlir::Operation*> ExpandRelayoutOp(RelayoutOp relayout,
                                            Layout target_layout,
                                            Layout input_layout,
                                            Layout output_layout) {
  bool match_present = false;
  for (const std::string& sharding_spec : target_layout.sharding_spec_strs())
    if (sharding_spec == Layout::kMatch) match_present = true;

  if (!match_present && output_layout != target_layout)
    return absl::InternalError(
        "output layout of Relayout op after layout propagation does not match "
        "layout specified by Relayout op.");
  auto value_or_status =
      EmitRelayout(relayout.getInput(), input_layout, output_layout);
  if (!value_or_status.ok())
    return absl::InvalidArgumentError(
        toolchain::formatv("Unsupported layout received for {0} op. Trying "
                      "to set tensor "
                      "to layout : {1}. Found error {2}",
                      relayout->getName().getStringRef(),
                      target_layout.ToString(),
                      value_or_status.status().message())
            .str());
  mlir::Value output = value_or_status.value();
  relayout.getOutput().replaceAllUsesWith(output);
  relayout.erase();
  return output.getDefiningOp();
}

// Takes relayout which may have kMatch dimensions and uses it to mask input.
// Here source_layout
StatusOr<Layout> MergeLayouts(
    const absl::flat_hash_set<std::string>& used_mesh_dimensions,
    const Layout& mask_layout, const Layout& target_layout) {
  std::vector<std::string> sharding_specs(mask_layout.sharding_spec_strs());
  for (int i = 0; i < target_layout.rank(); ++i) {
    if (sharding_specs[i] == Layout::kMatch &&
        !used_mesh_dimensions.contains(target_layout.sharding_spec(i)))
      sharding_specs[i] = target_layout.sharding_spec(i);
  }
  return Layout::GetLayout(sharding_specs, target_layout.mesh());
}

// Computes the layout of Relayout's (or RelayoutLike's) input or output, based
// on the layout from the corresponding output or input (as `incoming_layout`).
// Note that this implies that we compute the same layout for the
// operand and output.
// `mask_layout` is set to the user-supplied layout attribute on the op.
StatusOr<toolchain::DenseMap<int, Layout>> ComputeRelayoutLayout(
    const Layout& mask_layout, std::optional<const Layout> incoming_layout) {
  absl::flat_hash_set<std::string> used_dimensions;
  bool match_present = false;
  for (const std::string& sharding_spec : mask_layout.sharding_spec_strs()) {
    if (sharding_spec == Layout::kMatch)
      match_present = true;
    else if (Layout::IsShardedDimension(sharding_spec))
      used_dimensions.insert(sharding_spec);
  }
  if (!match_present) {
    return toolchain::DenseMap<int, Layout>({{0, mask_layout}});
  }

  if (incoming_layout) {
    TF_ASSIGN_OR_RETURN(
        Layout new_layout,
        MergeLayouts(used_dimensions, mask_layout, *incoming_layout));
    return toolchain::DenseMap<int, Layout>({{0, new_layout}});
  }
  return toolchain::DenseMap<int, Layout>();
}

}  // namespace

StatusOr<mlir::Operation*> RelayoutSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto relayout = mlir::cast<mlir::TF::RelayoutOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout target_layout,
                      Layout::FromString(relayout.getLayout().str()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(relayout.getInput()));

  return ExpandRelayoutOp<mlir::TF::RelayoutOp>(relayout, target_layout,
                                                input_layout, output_layout);
}

StatusOr<toolchain::DenseMap<int, Layout>>
RelayoutSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  auto relayout = toolchain::cast<mlir::TF::RelayoutOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout mask_layout,
                      Layout::FromString(relayout.getLayout().str()));
  std::optional<const Layout> incoming_layout;
  if (input_layouts.find(0) != input_layouts.end())
    incoming_layout.emplace(input_layouts.lookup(0));

  return ComputeRelayoutLayout(mask_layout, incoming_layout);
}

StatusOr<toolchain::DenseMap<int, Layout>>
RelayoutSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  auto relayout = toolchain::cast<mlir::TF::RelayoutOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout mask_layout,
                      Layout::FromString(relayout.getLayout().str()));
  std::optional<const Layout> incoming_layout;
  if (output_layouts.find(0) != output_layouts.end())
    incoming_layout.emplace(output_layouts.lookup(0));

  return ComputeRelayoutLayout(mask_layout, incoming_layout);
}

StatusOr<mlir::Operation*> RelayoutLikeSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto relayout_grad = mlir::cast<mlir::TF::RelayoutLikeOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout target_layout,
      ExtractRequiredLayoutFromOperand(relayout_grad.getLayoutInput()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(relayout_grad.getInput()));

  return ExpandRelayoutOp<mlir::TF::RelayoutLikeOp>(
      relayout_grad, target_layout, input_layout, output_layout);
}

StatusOr<toolchain::DenseMap<int, Layout>>
RelayoutLikeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // RelayoutLike's output has the same layout as the corresponding Relayout's
  // input operand.
  if (input_layouts.find(1) == input_layouts.end())
    return toolchain::DenseMap<int, Layout>();
  return toolchain::DenseMap<int, Layout>({{0, input_layouts.lookup(1)}});
}

StatusOr<toolchain::DenseMap<int, Layout>>
RelayoutLikeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  const Layout output_layout = output_layouts.lookup(0);
  return toolchain::DenseMap<int, Layout>({
      // Return replicated layout for the input operand since we do not want to
      // enforce any particular layout on it.
      {0, Layout::ReplicatedLike(output_layout)},
      // Set layout for the forward pass's input operand to match the output of
      // the RelayoutLike op.
      {1, output_layout},
  });
}

StatusOr<mlir::Operation*> DTensorSendSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  auto dtensor_send = toolchain::cast<mlir::TF::DTensorSend>(op);

  TF_ASSIGN_OR_RETURN(mlir::Operation * recv_op,
                      GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorSend>(
                          module, dtensor_send));
  auto dtensor_recv = toolchain::dyn_cast<mlir::TF::DTensorRecv>(recv_op);

  TF_RETURN_IF_ERROR(
      ValidateSendRecvLayoutConfiguration(dtensor_send, dtensor_recv));

  return LowerDTensorSend(op, recv_op);
}

// DTensorSend op respects input layout from input operations and does not
// set any preferred inputs layouts. During SPMD expansion, however, tensor
// values are changed to replicated layout before transferring data across mesh
// cluster.
StatusOr<toolchain::DenseMap<int, Layout>>
DTensorSendSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<toolchain::DenseMap<int, Layout>>
DTensorSendSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<mlir::Operation*> DTensorRecvSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  auto dtensor_recv = toolchain::cast<mlir::TF::DTensorRecv>(op);

  TF_ASSIGN_OR_RETURN(mlir::Operation * send_op,
                      GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorRecv>(
                          module, dtensor_recv));
  auto dtensor_send = toolchain::dyn_cast<mlir::TF::DTensorSend>(send_op);

  TF_RETURN_IF_ERROR(
      ValidateSendRecvLayoutConfiguration(dtensor_send, dtensor_recv));

  return LowerDTensorRecv(send_op, op);
}

// DTensorRecv always returns tensors with fully replicated layout.
StatusOr<toolchain::DenseMap<int, Layout>>
DTensorRecvSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  mlir::TF::DTensorRecv dtensor_recv =
      mlir::dyn_cast<mlir::TF::DTensorRecv>(op);
  if (!dtensor_recv) {
    return absl::InvalidArgumentError(
        toolchain::formatv("Expecting DTensorRecvOp but got {0}", OpName(op)).str());
  }
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<toolchain::DenseMap<int, Layout>>
DTensorRecvSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  return toolchain::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace machina
