/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This transformation forms clusters from instructions in same island and
// assigned to save devices. Clusters are represented as regions.
// Note that side-effecting ops are not correctly handled yet.

#include <memory>
#include <string>

#include "toolchain/ADT/MapVector.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/IRMapping.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/side_effect_analysis.h"
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/utils/cluster_util.h"
#include "machina/core/platform/logging.h"

namespace mlir {
namespace TFDevice {

namespace {

#define GEN_PASS_DEF_CLUSTERFORMATIONPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct ClusterFormationPass
    : public impl::ClusterFormationPassBase<ClusterFormationPass> {
  void runOnOperation() override;
};

void ReplaceLiveOutExternalUses(toolchain::ArrayRef<Value> live_outs,
                                tf_device::LaunchOp launch_op) {
  Region* launch_op_region = &launch_op.getBody();
  for (const auto& p : toolchain::zip(live_outs, launch_op.getResults())) {
    Value from = std::get<0>(p);
    // TODO(jingpu): move this to RegionUtils.h in MLIR core.
    for (auto& use : toolchain::make_early_inc_range(from.getUses())) {
      if (launch_op_region->isAncestor(use.getOwner()->getParentRegion()))
        continue;
      use.set(std::get<1>(p));
    }
  }
}

// Get all escaped live-out values of a region.
void GetLiveOuts(Region* region, toolchain::SmallVectorImpl<Value>* live_outs) {
  live_outs->clear();

  for (Operation& op : region->front()) {
    for (Value v : op.getResults()) {
      // A value is live-out if any of its users are not inside value producer's
      // region.
      bool is_live_out = toolchain::any_of(v.getUsers(), [&](Operation* user) {
        return !region->isAncestor(user->getParentRegion());
      });

      if (is_live_out) live_outs->emplace_back(v);
    }
  }
}

// Build a `tf_device.launch` op with a region that contains all the operations
// in given cluster. Then all ops in cluster are replaced by `tf_device.launch`.
void BuildLaunchForCluster(const TF::Cluster& c, OpBuilder* builder) {
  // Set insertion point to right after all operations in cluster.
  builder->setInsertionPoint(c.ops.back()->getNextNode());

  // Create a stand-alone region to hold all instructions in the cluster.
  Region region;
  region.push_back(new Block);

  // Move all operations in cluster to newly created region, stripping their
  // "device" attribute since launch op already carries device information.
  Block* block = &region.front();
  for (Operation* op : c.ops) {
    op->moveBefore(block, block->end());
    op->removeAttr(builder->getStringAttr("device"));
  }

  // Get all escaped live-out values of region, they are used later to determine
  // return values and types of launch op.
  toolchain::SmallVector<Value, 4> live_outs;
  GetLiveOuts(&region, &live_outs);

  // Build a `tf_device.return` op at end of region, with all live-out values
  // as operand.
  OpBuilder return_builder(builder->getContext());
  return_builder.setInsertionPointToEnd(block);
  return_builder.create<tf_device::ReturnOp>(return_builder.getUnknownLoc(),
                                             live_outs);

  toolchain::SmallVector<Type, 4> live_out_types;
  live_out_types.reserve(live_outs.size());
  for (Value v : live_outs) {
    live_out_types.emplace_back(v.getType());
  }

  tf_device::LaunchOp launch_op = builder->create<tf_device::LaunchOp>(
      builder->getUnknownLoc(), builder->getStringAttr(c.target),
      live_out_types);

  // Attach the region to launch_op.
  launch_op.getBody().takeBody(region);

  // Replace any external uses of live-out values with return values of launch
  // op. So live-out values no longer escape the region.
  ReplaceLiveOutExternalUses(live_outs, launch_op);

  // Ensure that users of the launch op's results appear after the launch op
  // in order to preserve the dominance property.
  TF::ReorderOpResultUses(launch_op);
}

std::string GetDevice(Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return device_attr ? device_attr.getValue().str() : "";
}

bool CanBeIgnoredInCluster(Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return !device_attr || device_attr.getValue().empty();
}

void BuildClusters(Block& block, OpBuilder builder,
                   const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  auto all_clusters = TF::BuildAllClusters(block, side_effect_analysis,
                                           GetDevice, CanBeIgnoredInCluster);
  for (const auto& [device, clusters] : all_clusters) {
    for (const auto& cluster : clusters) {
      BuildLaunchForCluster(cluster, &builder);
    }
  }
}

void ClusterFormationPass::runOnOperation() {
  auto module = getOperation();
  auto& side_effect_analysis = getAnalysis<TF::SideEffectAnalysis>();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    OpBuilder builder(func.getContext());
    const TF::SideEffectAnalysis::Info& info =
        side_effect_analysis.GetAnalysisForFunc(func);

    // Operates on individual blocks independently of if they are directly in
    // the function body or if they are nested in individual
    // `tf_executor.island`.
    for (Block& block : func.getBody()) BuildClusters(block, builder, info);
    func.walk([&](tf_executor::IslandOp island) {
      BuildClusters(island.GetBody(), builder, info);
    });
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateClusterFormationPass() {
  return std::make_unique<ClusterFormationPass>();
}

}  // namespace TFDevice
}  // namespace mlir
