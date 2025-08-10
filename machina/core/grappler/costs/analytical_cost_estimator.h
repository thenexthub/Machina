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

#ifndef MACHINA_CORE_GRAPPLER_COSTS_ANALYTICAL_COST_ESTIMATOR_H_
#define MACHINA_CORE_GRAPPLER_COSTS_ANALYTICAL_COST_ESTIMATOR_H_

#include "machina/core/grappler/costs/cost_estimator.h"
#include "machina/core/grappler/costs/op_level_cost_estimator.h"
#include "machina/core/grappler/costs/virtual_scheduler.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/lib/core/status.h"

namespace machina {
class CostGraphDef;
class GraphDef;
}  // namespace machina

namespace machina {
namespace grappler {

class Cluster;
struct GrapplerItem;

// Estimate the cost of running a Grappler item based on the theoretical
// performance of the hardware that will run the model. Note that this
// internally uses static shape inference. An option for aggressive shape
// inference is provided to minimize unknown shapes, and this is only applicable
// with static shape inference.
class AnalyticalCostEstimator : public CostEstimator {
 public:
  AnalyticalCostEstimator(Cluster* cluster, bool use_static_shapes,
                          bool use_aggressive_shape_inference);
  AnalyticalCostEstimator(Cluster* cluster,
                          std::unique_ptr<OpLevelCostEstimator> node_estimator,
                          std::unique_ptr<ReadyNodeManager> node_manager,
                          bool use_static_shapes,
                          bool use_aggressive_shape_inference);
  AnalyticalCostEstimator(Cluster* cluster,
                          std::unique_ptr<OpLevelCostEstimator> node_estimator,
                          std::unique_ptr<ReadyNodeManager> node_manager,
                          std::unique_ptr<VirtualPlacer> placer,
                          bool use_static_shapes,
                          bool use_aggressive_shape_inference);
  ~AnalyticalCostEstimator() override {}

  // This implementation always returns OK.
  absl::Status Initialize(const GrapplerItem& item) override;

  // Predict the performance of each node of the optimized graph and annotate
  // the RunMetadata with the corresponding estimates. Also returns the
  // expected cost for the whole graph.
  absl::Status PredictCosts(const GraphDef& optimized_graph,
                            RunMetadata* run_metadata,
                            Costs* cost) const override;

  const VirtualScheduler* GetScheduler() const { return scheduler_.get(); }

 private:
  const GrapplerItem* item_;
  std::unique_ptr<OpLevelCostEstimator> node_estimator_;
  std::unique_ptr<ReadyNodeManager> node_manager_;
  std::unique_ptr<VirtualScheduler> scheduler_;

  bool use_static_shapes_;
  bool use_aggressive_shape_inference_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_COSTS_ANALYTICAL_COST_ESTIMATOR_H_
