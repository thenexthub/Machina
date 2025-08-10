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

#ifndef MACHINA_PYTHON_GRAPPLER_COST_ANALYZER_H_
#define MACHINA_PYTHON_GRAPPLER_COST_ANALYZER_H_

#include <iostream>
#include "machina/core/framework/cost_graph.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/grappler/costs/analytical_cost_estimator.h"
#include "machina/core/grappler/costs/cost_estimator.h"
#include "machina/core/grappler/costs/measuring_cost_estimator.h"
#include "machina/core/grappler/costs/op_performance_data.pb.h"

namespace machina {
class GraphDef;
class CostGraphDef;

namespace grappler {
struct GrapplerItem;

// Aggregated perf summary for ops of the same type in a graph.
struct OpPerfSummary {
  string name;
  int64_t count;
  int64_t time;
  int64_t compute_time;
  int64_t memory_time;
  // Upper and lower bound for estimated time.
  int64_t time_upper;
  int64_t time_lower;
};

// Generate op-level performance insights on compute/memory
// efficiency, as well as graph-level aggregated performance statistics.
class CostAnalyzer {
 public:
  explicit CostAnalyzer(const GrapplerItem& item, Cluster* cluster,
                        const string& suffix);
  absl::Status GenerateReport(std::ostream& os, bool per_node_report,
                              bool verbose);

 private:
  void PredictCosts(CostEstimator* cost_estimator, CostGraphDef* cost_graph,
                    int64_t* total_time);
  void GatherCosts();
  void PreprocessCosts();
  void AnalyzeCosts();
  void SortOpsByTime(std::map<string, OpPerfSummary> ops);
  void PrintAnalysis(std::ostream& os, bool per_node_report,
                     bool verbose) const;

  const GrapplerItem* item_;
  MeasuringCostEstimator measure_estimator_;
  AnalyticalCostEstimator analytical_estimator_;
  OpPerformanceList op_perf_;
  OpPerformanceList op_perf_analytical_;
  int64_t total_time_measured_;
  int64_t total_time_analytical_;
  std::vector<OpPerfSummary> ops_;
  int64_t total_time_measured_serialized_;
  int64_t total_time_analytical_upper_;
  int64_t total_time_analytical_lower_;
  string suffix_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_PYTHON_GRAPPLER_COST_ANALYZER_H_
