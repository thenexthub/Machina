/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_

#include <unordered_set>

#include "machina/core/grappler/costs/graph_properties.h"
#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/grappler/utils.h"
#include "machina/core/grappler/utils/frame.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace grappler {

constexpr char kLoopOptimizer[] = "LoopOptimizer";

class LoopOptimizer : public GraphOptimizer {
 public:
  LoopOptimizer();

  explicit LoopOptimizer(RewriterConfig::Toggle opt_level,
                         DeviceBase* cpu_device);

  ~LoopOptimizer() override {}

  string name() const override { return "loop_optimizer"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

 private:
  friend class LoopOptimizerTest;

  // Granular control for loop optimizer stages.
  struct LoopOptimizerOptions {
    bool enable_loop_invariant_node_motion = false;
    bool enable_stack_push_removal = true;
    bool enable_dead_branch_removal = true;

    static LoopOptimizerOptions Default(RewriterConfig::Toggle opt_level) {
      LoopOptimizerOptions options;
      return options;
    }
  };

  absl::Status RemoveDeadBranches(
      const std::unordered_set<string>& nodes_to_preserve, NodeMap& node_map,
      const absl::flat_hash_set<string>& feed_nodes, GraphDef* optimized_graph);

  RewriterConfig::Toggle opt_level_;
  DeviceBase* cpu_device_;
  LoopOptimizerOptions options_;
  std::unique_ptr<ResourceMgr> resource_mgr_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_
