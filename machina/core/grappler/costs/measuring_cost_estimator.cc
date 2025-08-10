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

#include "machina/core/grappler/costs/measuring_cost_estimator.h"

#include <limits>

#include "machina/core/framework/cost_graph.pb.h"
#include "machina/core/framework/step_stats.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/grappler/costs/robust_stats.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/platform/blocking_counter.h"
#include "machina/core/platform/env.h"
#include "machina/core/public/session.h"

namespace machina {
namespace grappler {

MeasuringCostEstimator::MeasuringCostEstimator(Cluster* cluster,
                                               int measurement_steps,
                                               int measurement_threads)
    : measurement_steps_(measurement_steps),
      measurement_threads_(measurement_threads) {
  CHECK_GE(measurement_steps, 1);
  if (measurement_threads > 0) {
    thread_pool_.reset(new thread::ThreadPool(
        Env::Default(), SanitizeThreadSuffix("measurements"),
        measurement_threads));
  }
  cluster_ = cluster;
}

absl::Status MeasuringCostEstimator::Initialize(const GrapplerItem& item) {
  feed_ = item.feed;
  fetch_ = item.fetch;
  return cluster_->Initialize(item);
}

absl::Status MeasuringCostEstimator::PredictCosts(
    const GraphDef& optimized_graph, RunMetadata* run_metadata,
    Costs* costs) const {
  CostGraphDef* cost_graph = nullptr;
  if (run_metadata) {
    cost_graph = run_metadata->mutable_cost_graph();
  }
  const bool running_simulation = (cluster_->type() == "virtual");

  std::vector<double> times(measurement_steps_);
  BlockingCounter barrier(measurement_steps_);

  mutex status_mu;
  absl::Status status;

  auto measurement_fn = [&](const int step) {
    const Costs::MicroSeconds start = Env::Default()->NowMicros();

    RunMetadata metadata;
    const absl::Status local_status =
        cluster_->Run(optimized_graph, feed_, fetch_, &metadata);
    {
      mutex_lock lock(status_mu);
      status.Update(local_status);
    }
    if (step < 0) {
      // Discard the first iteration as it triggers the warmup, and therefore
      // takes much longer than a normal step.
      return;
    }
    if (!local_status.ok()) {
      // Discard the data if the run wasn't successful.
      barrier.DecrementCount();
      return;
    }

    const Costs::MicroSeconds finish = Env::Default()->NowMicros();
    if (running_simulation) {
      // When running simulation, return the estimated runtime, not the time it
      // takes to run the simulation.
      double time = 0.0;
      for (const DeviceStepStats& stepstats :
           metadata.step_stats().dev_stats()) {
        for (const NodeExecStats& node_stats : stepstats.node_stats()) {
          const double completion_time =
              node_stats.all_end_rel_micros() + node_stats.all_start_micros();
          time = std::max(time, completion_time * 1e3);
        }
      }
      times[step] = time;
    } else {
      const double time = (finish - start).count() * 1e3;
      times[step] = time;
    }
    if (cost_graph && (step + 1 == measurement_steps_)) {
      metadata.mutable_cost_graph()->Swap(cost_graph);
    }

    barrier.DecrementCount();
  };

  // Initialize the computation and warm up TensorFlow.
  measurement_fn(-1);

  if (!status.ok()) {
    LOG(ERROR) << "Failed to run start measurements: " << status.message();
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  // Run "measurement_steps_" and measure the time.
  VLOG(1) << "Number of measurement steps: " << measurement_steps_;
  if (measurement_threads_ > 0) {
    for (int i = 0; i < measurement_steps_; ++i) {
      thread_pool_->Schedule([i, &measurement_fn]() { measurement_fn(i); });
    }
    barrier.Wait();
  } else {
    for (int i = 0; i < measurement_steps_ && status.ok(); ++i) {
      measurement_fn(i);
    }
  }

  if (!status.ok()) {
    LOG(ERROR) << "Failed to measure graph performance: " << status.message();
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  // Compute the average time of the measure steps. Use Huber statistics
  // to filter out outliers.
  RobustStats stats(times);
  costs->execution_time = Costs::Duration(stats.mean());

  return absl::OkStatus();
}
}  // end namespace grappler
}  // end namespace machina
