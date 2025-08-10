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

#ifndef MACHINA_CORE_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_
#define MACHINA_CORE_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "machina/cc/training/coordinator.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/cost_graph.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/lib/core/threadpool.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/queue_runner.pb.h"
#include "machina/core/public/session.h"

namespace machina {
namespace grappler {

// Create a simple cluster that makes available to grappler a subset of the
// nodes available on a single local computer.
class SingleMachine : public Cluster {
 public:
  SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus);
  ~SingleMachine() override;

  string type() const override { return "single_machine"; }

  absl::Status Provision() override;
  absl::Status Shutdown() override;

  absl::Status Initialize(const GrapplerItem& item) override;
  absl::Status Run(const GraphDef& item,
                   const std::vector<std::pair<string, Tensor>>& feed,
                   const std::vector<string>& fetch,
                   RunMetadata* metadata) override;

  const DeviceSet* GetDeviceSet() const override { return device_set_.get(); }

  absl::Status EnablePeakMemoryStats() override;

  // It requires EnableAllocatorStats(true) be called before Provision().
  absl::Status GetPeakMemoryUsage(
      std::unordered_map<string, uint64>* device_peak_memory) const override;

 private:
  absl::Status RunWithTimeout(
      const std::vector<std::pair<string, Tensor>>& feed,
      const std::vector<string>& fetch, RunMetadata* run_metadata);
  absl::Status RunWithTimeout(
      const std::vector<std::pair<string, Tensor>>& feed,
      const std::vector<string>& fetch, RunMetadata* run_metadata,
      int64_t timeout_s);
  absl::Status ResetSession();
  absl::Status CloseSession(bool use_timeout);
  absl::Status ShutdownSession();
  void MergeCosts(CostGraphDef* graph_costs, const CostGraphDef& init_costs,
                  const CostGraphDef& queue_costs);

  absl::Status ClearAllocatorStats() const;

  std::unique_ptr<Session> session_;
  std::vector<QueueRunnerDef> queue_runner_defs_;
  string last_graph_id_;
  mutex last_graph_mu_;
  const GraphDef* last_graph_ TF_GUARDED_BY(last_graph_mu_) = nullptr;
  std::vector<string> init_ops_;
  int64_t expected_init_time_s_;
  std::unique_ptr<Coordinator> coordinator_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<DeviceSet> device_set_;

  RunMetadata init_metadata_;

  mutex close_mu_;
  bool closing_ TF_GUARDED_BY(close_mu_);

  bool cpu_allocator_stats_enabled_ = false;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_CLUSTERS_SINGLE_MACHINE_H_
