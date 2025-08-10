/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_

#include "machina/core/framework/variable.pb.h"
#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace grappler {

// Automatically parallelize a graph by splitting in the batch dimension.
class AutoParallel : public GraphOptimizer {
 public:
  AutoParallel(int num_replicas) : num_replicas_(num_replicas) {
    CHECK(num_replicas_ >= 2);
  }
  ~AutoParallel() override {}

  string name() const override { return "autoparallel"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;

 private:
  GraphDef graph_;
  std::map<string, NodeDef*> all_nodes_;
  std::set<string> apply_gradients_nodes_;
  std::set<string> replica_nodes_;
  std::set<string> shared_nodes_;
  const GrapplerItem* item_;
  int num_replicas_;
  int num_gpus_;
  absl::Status Initialize(const GrapplerItem& item);
  NodeDef* AddNodeDivConst();
  NodeDef* AddNodeDiv(const string& name, const string& input_a,
                      const string& input_b);
  NodeDef* AddNodeControl(const string& name, const std::set<string>& deps,
                          GraphDef* graph);
  bool NotSharedNode(const string& name);
  void AddSharedNodes(GraphDef* graph);
  void AddOneReplica(GraphDef* graph, int number);
  void BuildGraph(GraphDef* graph);
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_
