/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_COMMON_SUBGRAPH_ELIMINATION_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_COMMON_SUBGRAPH_ELIMINATION_H_

#include <unordered_set>

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/lib/gtl/flatset.h"
#include "machina/core/platform/hash.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace grappler {

// Optimize TF computations by deduping equivalent subgraphs.
class Cluster;
struct GrapplerItem;

class CommonSubgraphElimination : public GraphOptimizer {
 public:
  CommonSubgraphElimination() {}

  explicit CommonSubgraphElimination(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}

  ~CommonSubgraphElimination() override {}

  string name() const override { return "common_subgraph_elimination"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

 private:
  friend class CommonSubgraphEliminationTest;

  // Returns true if it is safe to dedup node from the graph.
  bool CanDedup(const NodeDef& node) const;

  // Dedup redundant nodes in the graph.
  absl::Status DedupComputations(GraphDef* optimized_graph);

  RewriterConfig::Toggle opt_level_;

  bool fetch_nodes_known_ = false;
  std::unordered_set<string> nodes_to_preserve_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_COMMON_SUBGRAPH_ELIMINATION_H_
