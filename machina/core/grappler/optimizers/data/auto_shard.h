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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_

#include <string>
#include <vector>

#include "machina/core/framework/dataset_options.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/mutable_graph_view.h"
#include "machina/core/grappler/optimizers/data/optimizer_base.h"

namespace machina {
namespace grappler {

class AutoShard : public TFDataOptimizerBase {
 public:
  AutoShard() = default;
  ~AutoShard() override = default;

  string name() const override { return "tf_auto_shard"; }

  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Init(
      const machina::RewriterConfig_CustomGraphOptimizer* config) override;

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;

 private:
  int64_t num_workers_;
  int64_t num_replicas_;
  int64_t index_;
  machina::data::AutoShardPolicy auto_shard_policy_;
};

// For testing only
namespace internal {
bool IsEligibleRewriteBatchSize(const NodeDef& sink_node,
                                const MutableGraphView& graph,
                                std::vector<std::string>* ineligible_reason);
}

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_
