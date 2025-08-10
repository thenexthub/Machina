/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/mlir/machina/utils/cluster_util.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/side_effect_analysis.h"

namespace mlir::TF {

namespace {

// Exhaust search in the block to get all the ops that have data dependency of
// the cluster.
toolchain::SetVector<Operation*> GetAllOpsDependOnCluster(
    const Cluster& c,
    const toolchain::DenseMap<Operation*, Cluster*>& op_to_cluster_map) {
  toolchain::SetVector<Operation*> ops_depend_on_cluster;
  for (Operation& op : *c.ops.front()->getBlock()) {
    if (op.isBeforeInBlock(c.ops.front()) || c.ops.contains(&op)) {
      continue;
    }
    // Gets the live in values of the `op`
    toolchain::SetVector<Value> live_ins(op.operand_begin(), op.operand_end());
    getUsedValuesDefinedAbove(op.getRegions(), live_ins);
    // Inserts if any of the `live_ins` depends on the ops in the cluster.
    if (toolchain::any_of(live_ins, [&](Value value) {
          Operation* defining_op = value.getDefiningOp();
          if (!defining_op) {
            return false;
          }
          return c.ops.contains(defining_op) ||
                 ops_depend_on_cluster.contains(defining_op);
        })) {
      ops_depend_on_cluster.insert(&op);
    }
  }
  // The data dependency of the cluster includes the union of ops' data
  // dependency. So includes all the ops in the same cluster of the op in
  // `ops_depend_on_cluster`.
  toolchain::SetVector<Operation*> same_cluster_ops_with_dependency(
      ops_depend_on_cluster.begin(), ops_depend_on_cluster.end());
  for (Operation* op : ops_depend_on_cluster) {
    Cluster* cluster = op_to_cluster_map.lookup(op);
    if (cluster == nullptr) {
      continue;
    }
    for (Operation* ops_in_same_cluster : cluster->ops) {
      same_cluster_ops_with_dependency.insert(ops_in_same_cluster);
    }
  }
  return same_cluster_ops_with_dependency;
}

// An op can be merged into cluster if it satisfies both of the following
// conditions:
//
//  * Merging the op into the cluster doesn't break the acyclic nature of the
//  *   graph. This means all of its operands don't have data dependency of the
//  *   cluster.
//  * Merging the op into the cluster does not reorder control dependencies.
//
bool CanMergeIntoCluster(
    const Cluster& c, Operation* to_merge,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    const toolchain::DenseMap<Operation*, Cluster*>& op_to_cluster_map) {
  // If any of the op's control predecessors appears after the last op in the
  // cluster, merging the op may cause control dependencies to be reordered.
  // Hence, the op cannot be merged to the cluster in such a case.
  const bool has_control_predecessors_after_cluster =
      !side_effect_analysis
           .DirectControlPredecessors(
               to_merge,
               [&c](Operation* pred) {
                 Operation* const last_c_op = c.ops.back();
                 return last_c_op->getBlock() == pred->getBlock() &&
                        last_c_op->isBeforeInBlock(pred);
               })
           .empty();
  if (has_control_predecessors_after_cluster) {
    return false;
  }

  // We can merge the op into the cluster if doing so doesn't break the acyclic
  // nature of the graph. In this way, we need to check if there is any
  // dependency from the oprands of the op to the current cluster.
  toolchain::SetVector<Operation*> ops_depend_on_cluster =
      GetAllOpsDependOnCluster(c, op_to_cluster_map);
  return toolchain::none_of(to_merge->getOperands(), [&](Value value) {
    Operation* defining_op = value.getDefiningOp();
    return defining_op && ops_depend_on_cluster.contains(defining_op);
  });
}
}  // namespace

toolchain::StringMap<SmallVector<Cluster>> BuildAllClusters(
    Block& block, const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    std::function<bool(Operation*)> is_ignored_op) {
  // Iteratively find clusters of different targets within the `block`.
  // Whenever we see an operation that is assigned to an accelerator target
  // (ie. get_target(op) != ""), we try to merge it into the last cluster
  // of same target. If that is infeasible (say because of violating
  // def-before-use), create a new cluster with that operation and move on.
  toolchain::StringMap<SmallVector<Cluster>> all_clusters;
  // Map from operation to the cluster that contains the operation.
  toolchain::DenseMap<Operation*, Cluster*> op_to_cluster_map;

  toolchain::StringMap<Cluster> nearest_clusters;
  for (Operation& op : toolchain::make_early_inc_range(block)) {
    if (is_ignored_op(&op)) {
      continue;
    }
    std::string target_name = get_target(&op);

    // If no cluster of same target has been formed yet, create a new cluster
    // with op alone.
    auto it = nearest_clusters.find(target_name);
    if (it == nearest_clusters.end()) {
      SetVector<Operation*> new_cluster_op_set;
      new_cluster_op_set.insert(&op);
      nearest_clusters[target_name] = Cluster{new_cluster_op_set, target_name};
      op_to_cluster_map[&op] = &nearest_clusters[target_name];
      continue;
    }

    // Check if it is legal to merge op into nearest cluster of same target.
    // If positive, update cluster and move on to next operation.
    Cluster& nearest_cluster = it->second;
    if (CanMergeIntoCluster(nearest_cluster, &op, side_effect_analysis,
                            get_target, op_to_cluster_map)) {
      nearest_cluster.ops.insert(&op);
      op_to_cluster_map[&op] = &nearest_cluster;
      continue;
    }

    // If nearest cluster of same target can not absorb `op`, then that
    // cluster needs to be finalized by inserting into the final cluster map
    // that contains all operations in clusters.
    all_clusters[target_name].push_back(nearest_cluster);

    // Create a new cluster to hold op alone and update nearest_clusters.
    SetVector<Operation*> new_cluster_op_set;
    new_cluster_op_set.insert(&op);
    nearest_clusters[target_name] = Cluster{new_cluster_op_set, target_name};
    op_to_cluster_map[&op] = &nearest_clusters[target_name];
  }

  // At the end, there might be left-over found clusters that need to be
  // built.
  for (auto& target_cluster : nearest_clusters) {
    all_clusters[target_cluster.first()].push_back(target_cluster.second);
  }

  return all_clusters;
}

void ReorderOpResultUses(mlir::Operation* cluster) {
  mlir::Block* const cluster_block = cluster->getBlock();
  toolchain::SetVector<mlir::Operation*> ops_to_reorder;

  toolchain::SmallVector<mlir::Value> worklist;
  toolchain::append_range(worklist, cluster->getResults());

  while (!worklist.empty()) {
    mlir::Value value = worklist.back();
    worklist.pop_back();

    for (mlir::Operation* const user : value.getUsers()) {
      mlir::Operation* const op = cluster_block->findAncestorOpInBlock(*user);
      if (op == nullptr || !op->isBeforeInBlock(cluster)) {
        continue;
      }

      if (ops_to_reorder.insert(op)) {
        toolchain::append_range(worklist, op->getResults());
      }
    }
  }

  toolchain::SmallVector<mlir::Operation*, 0> sorted = ops_to_reorder.takeVector();
  toolchain::sort(sorted, [](mlir::Operation* lhs, mlir::Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  for (mlir::Operation* const op : toolchain::reverse(sorted)) {
    op->moveAfter(cluster);
  }
}

}  // namespace mlir::TF
