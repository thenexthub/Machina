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

#include "machina/core/grappler/optimizers/data/parallel_batch.h"

#include "absl/status/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/mutable_graph_view.h"
#include "machina/core/grappler/op_types.h"
#include "machina/core/grappler/optimizers/custom_graph_optimizer_registry.h"

namespace machina {
namespace grappler {

absl::Status ParallelBatch::OptimizeAndCollectStats(Cluster* cluster,
                                                    const GrapplerItem& item,
                                                    GraphDef* output,
                                                    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == "BatchDatasetV2" || node.op() == "PaddedBatchDatasetV2") {
      (*node.mutable_attr())["parallel_copy"].set_b(true);
      stats->num_changes++;
    }
  }
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(ParallelBatch, "parallel_batch");

}  // namespace grappler
}  // namespace machina
