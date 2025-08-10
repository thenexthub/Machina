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

#include "machina/core/grappler/clusters/virtual_cluster.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/framework/cost_graph.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/grappler/clusters/utils.h"
#include "machina/core/grappler/costs/op_level_cost_estimator.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/device_properties.pb.h"

namespace machina {
namespace grappler {

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices)
    : VirtualCluster(devices, std::make_unique<OpLevelCostEstimator>(),
                     ReadyNodeManagerFactory("FirstReady")) {}

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices,
    std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager)
    : Cluster(0) {
  devices_ = devices;

  // Note that we do not use aggressive shape inference to preserve unknown
  // shapes from the input graph.
  estimator_ = std::make_unique<AnalyticalCostEstimator>(
      this, std::move(node_estimator), std::move(node_manager),
      /*use_static_shapes=*/true, /*use_aggressive_shape_inference=*/false);
}

VirtualCluster::VirtualCluster(const DeviceSet* device_set)
    : VirtualCluster(std::unordered_map<string, DeviceProperties>()) {
  device_set_ = device_set;
  for (const auto& device : device_set_->devices()) {
    DeviceProperties props = GetDeviceInfo(device->parsed_name());
    if (props.type() == "UNKNOWN") continue;
    auto attrs = device->attributes();
    props.set_memory_size(attrs.memory_limit());
    devices_[device->name()] = props;
  }
}

VirtualCluster::~VirtualCluster() {}

absl::Status VirtualCluster::Provision() { return absl::OkStatus(); }

absl::Status VirtualCluster::Initialize(const GrapplerItem& item) {
  return absl::OkStatus();
}

absl::Status VirtualCluster::Run(
    const GraphDef& graph, const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* metadata) {
  GrapplerItem item;
  item.graph = graph;
  item.feed = feed;
  item.fetch = fetch;
  return Run(item, metadata);
}

absl::Status VirtualCluster::Run(const GrapplerItem& item,
                                 RunMetadata* metadata) {
  // Initializes an analytical cost estimator to estimate the graph cost. Makes
  // sure to use static shape inference to prevent the virtual scheduler from
  // calling the Run method on the cluster and creating an infinite loop.
  if (metadata) {
    metadata->clear_step_stats();
    metadata->clear_cost_graph();
    metadata->clear_partition_graphs();
  }

  TF_RETURN_IF_ERROR(estimator_->Initialize(item));
  TF_RETURN_IF_ERROR(
      estimator_->PredictCosts(item.graph, metadata, /*cost=*/nullptr));

  const std::unordered_map<string, DeviceProperties>& device = GetDevices();
  std::unordered_map<string, int64_t> peak_mem_usage =
      estimator_->GetScheduler()->GetPeakMemoryUsage();
  for (const auto& mem_usage : peak_mem_usage) {
    const string& device_name = mem_usage.first;
    auto it = device.find(device_name);
    if (it == device.end()) {
      // It's probably the fake send/recv device. Eventually we'll need to
      // remove this fake device to ensure proper memory accounting for
      // multi-device settings.
      continue;
    }
    const DeviceProperties& dev = it->second;
    if (dev.memory_size() <= 0) {
      // Available device memory unknown
      continue;
    }
    int64_t peak_mem = mem_usage.second;
    if (peak_mem >= dev.memory_size()) {
      return errors::ResourceExhausted(
          "Graph requires ", peak_mem, " bytes of memory on device ",
          device_name, " to run ", " but device only has ", dev.memory_size(),
          " available.");
    }
  }

  return absl::OkStatus();
}

}  // namespace grappler
}  // namespace machina
