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

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/core/framework/kernel_def.pb.h"
#include "machina/core/framework/memory_types.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/step_stats.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/grappler/clusters/single_machine.h"
#include "machina/core/grappler/clusters/virtual_cluster.h"
#include "machina/core/grappler/costs/cost_estimator.h"
#include "machina/core/grappler/costs/graph_memory.h"
#include "machina/core/grappler/costs/measuring_cost_estimator.h"
#include "machina/core/grappler/costs/op_level_cost_estimator.h"
#include "machina/core/grappler/costs/op_performance_data.pb.h"
#include "machina/core/grappler/costs/utils.h"
#include "machina/core/grappler/devices.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/utils.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/device_properties.pb.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

absl::Status _GetOpPerformanceDataAndRunTime(
    const machina::grappler::GrapplerItem& item,
    machina::grappler::CostEstimator* cost_measure,
    machina::OpPerformanceList* op_performance_data,
    machina::grappler::Costs* costs) {
  absl::Status status = cost_measure->Initialize(item);
  if (!status.ok()) return status;

  machina::RunMetadata run_metadata;
  tsl::MaybeRaiseRegisteredFromStatus(
      cost_measure->PredictCosts(item.graph, &run_metadata, costs));

  if (op_performance_data) {
    *op_performance_data = machina::grappler::CostGraphToOpPerformanceData(
        run_metadata.cost_graph(), item.graph);
  }
  return absl::OkStatus();
}

PYBIND11_MAKE_OPAQUE(machina::grappler::Cluster);

PYBIND11_MODULE(_pywrap_tf_cluster, m) {
  py::class_<machina::grappler::Cluster> grappler_cluster(m, "Cluster");

  m.def("TF_NewCluster",
        [](bool allow_soft_placement,
           bool disable_detailed_stats) -> machina::grappler::Cluster* {
          // TODO(petebu): Make these named arguments with default values
          // instead.
          int num_cpu_cores =
              machina::grappler::GetNumAvailableLogicalCPUCores();
          int num_gpus = machina::grappler::GetNumAvailableGPUs();
          int timeout_s = 60 * 10;
          std::unique_ptr<machina::grappler::Cluster> cluster =
              std::make_unique<machina::grappler::SingleMachine>(
                  timeout_s, num_cpu_cores, num_gpus);
          cluster->DisableDetailedStats(disable_detailed_stats);
          cluster->AllowSoftPlacement(allow_soft_placement);
          cluster->SetNumWarmupSteps(10);
          tsl::MaybeRaiseRegisteredFromStatus(cluster->Provision());
          return cluster.release();
        });

  m.def("TF_NewVirtualCluster",
        [](const std::vector<py::bytes>& serialized_named_devices)
            -> machina::grappler::Cluster* {
          std::vector<machina::NamedDevice> named_devices;
          for (const auto& s : serialized_named_devices) {
            machina::NamedDevice named_device;
            if (!named_device.ParseFromString(std::string(s))) {
              throw std::invalid_argument(
                  "The NamedDevice could not be parsed as a valid protocol "
                  "buffer");
            }
            named_devices.push_back(named_device);
          }

          std::unordered_map<std::string, machina::DeviceProperties> devices;
          for (const auto& named_device : named_devices) {
            devices[named_device.name()] = named_device.properties();
          }
          std::unique_ptr<machina::grappler::Cluster> cluster =
              std::make_unique<machina::grappler::VirtualCluster>(devices);
          {
            // TODO(petebu): Do we need to hold the GIL here?
            py::gil_scoped_acquire acquire;
            tsl::MaybeRaiseRegisteredFromStatus(cluster->Provision());
          }
          return cluster.release();
        });

  m.def("TF_ShutdownCluster", [](machina::grappler::Cluster* cluster) {
    // TODO(petebu): Do we need to hold the GIL here?
    py::gil_scoped_acquire acquire;
    (void)cluster->Shutdown();
  });

  m.def("TF_ListDevices",
        [](machina::grappler::Cluster* cluster) -> std::vector<py::bytes> {
          const std::unordered_map<std::string, machina::DeviceProperties>&
              devices = cluster->GetDevices();
          std::vector<py::bytes> named_devices;
          for (auto& dev : devices) {
            machina::NamedDevice d;
            d.set_name(dev.first);
            *d.mutable_properties() = dev.second;
            named_devices.push_back(d.SerializeAsString());
          }
          return named_devices;
        });

  m.def("TF_ListAvailableOps", []() -> std::vector<std::string> {
    machina::OpRegistry* registry = machina::OpRegistry::Global();
    std::vector<machina::OpDef> ops;
    registry->GetRegisteredOps(&ops);
    std::vector<std::string> op_names;
    op_names.reserve(ops.size());
    for (const machina::OpDef& op : ops) {
      op_names.push_back(op.name());
    }
    std::sort(op_names.begin(), op_names.end());
    return op_names;
  });

  m.def(
      "TF_GetSupportedDevices",
      [](machina::grappler::Cluster* cluster,
         machina::grappler::GrapplerItem* item)
          -> std::unordered_map<std::string, std::vector<std::string>> {
        if (cluster == nullptr || item == nullptr) {
          tsl::MaybeRaiseRegisteredFromStatus(absl::Status(
              machina::errors::Internal("You need both a cluster and an "
                                           "item to get supported devices.")));
        }
        const std::unordered_map<std::string, machina::DeviceProperties>&
            devices = cluster->GetDevices();
        std::unordered_map<std::string, std::vector<std::string>> device_types;
        for (const auto& dev : devices) {
          device_types[dev.second.type()].push_back(dev.first);
        }

        std::unordered_map<std::string, std::set<std::string>>
            supported_device_types;
        std::unordered_map<std::string, std::set<std::string>>
            device_restrictions;

        for (const auto& node : item->graph.node()) {
          for (const auto& dev : device_types) {
            const std::string& type = dev.first;
            if (cluster->type() != "single_machine") {
              // The actual kernel may not be linked in this binary.
              supported_device_types[node.name()].insert(type);
            } else {
              // Check the kernel capabilities
              const machina::DeviceType dev_type(type);
              absl::Status s =
                  machina::FindKernelDef(dev_type, node, nullptr, nullptr);
              if (s.ok()) {
                supported_device_types[node.name()].insert(type);

                // Check which inputs are restricted to reside on the host.
                // TODO: extends this to support outputs as well
                machina::MemoryTypeVector inp_mtypes;
                machina::MemoryTypeVector out_mtypes;
                absl::Status s = machina::MemoryTypesForNode(
                    machina::OpRegistry::Global(), dev_type, node,
                    &inp_mtypes, &out_mtypes);
                if (s.ok()) {
                  for (size_t i = 0; i < inp_mtypes.size(); ++i) {
                    if (inp_mtypes[i] == machina::HOST_MEMORY) {
                      device_restrictions[machina::grappler::NodeName(
                                              node.input(i))]
                          .insert("CPU");
                      break;
                    }
                  }
                }
              }
            }
          }
        }

        std::unordered_map<std::string, std::vector<std::string>> result;
        for (const auto& supported_dev : supported_device_types) {
          const std::string& node = supported_dev.first;
          std::set<std::string> feasible;
          const auto it = device_restrictions.find(node);
          if (it != device_restrictions.end()) {
            const std::set<std::string>& candidates = supported_dev.second;
            const std::set<std::string>& valid = it->second;
            std::set_intersection(candidates.begin(), candidates.end(),
                                  valid.begin(), valid.end(),
                                  std::inserter(feasible, feasible.begin()));
          } else {
            feasible = supported_dev.second;
          }

          std::vector<std::string> device_names;
          for (const std::string& type : feasible) {
            auto it = device_types.find(type);
            DCHECK(it != device_types.end());
            for (const std::string& name : it->second) {
              device_names.push_back(name);
            }
          }
          result[node] = device_names;
        }
        return result;
      });

  m.def("TF_EstimatePerformance", [](const py::bytes& serialized_device) {
    machina::NamedDevice device;
    if (!device.ParseFromString(std::string(serialized_device))) {
      throw std::invalid_argument(
          "The NamedDevice could not be parsed as a valid protocol buffer");
    }
    machina::grappler::OpLevelCostEstimator estimator;
    machina::grappler::DeviceInfo info =
        estimator.GetDeviceInfo(device.properties());
    return info.gigaops;
  });

  m.def("TF_MeasureCosts",
        [](machina::grappler::GrapplerItem* item,
           machina::grappler::Cluster* cluster, bool generate_timeline)
            -> std::tuple<std::vector<py::bytes>, double, py::bytes> {
          const int num_measurements = cluster->type() == "virtual" ? 1 : 10;
          machina::grappler::MeasuringCostEstimator cost_measure(
              cluster, num_measurements, 0);

          machina::OpPerformanceList op_performance_data;
          machina::grappler::Costs costs;
          absl::Status s = _GetOpPerformanceDataAndRunTime(
              *item, &cost_measure, &op_performance_data, &costs);
          double run_time = FLT_MAX;
          if (s.ok()) {
            run_time = static_cast<double>(costs.execution_time.count()) / 1e9;
          }
          machina::StepStats step_stats;
          if (generate_timeline) {
            machina::RunMetadata metadata;
            tsl::MaybeRaiseRegisteredFromStatus(
                cluster->Run(item->graph, item->feed, item->fetch, &metadata));
            step_stats = metadata.step_stats();
          }

          std::vector<py::bytes> op_perf_objs;
          op_perf_objs.resize(op_performance_data.op_performance_size());
          for (int i = 0; i < op_performance_data.op_performance_size(); i++) {
            op_perf_objs[i] =
                op_performance_data.op_performance(i).SerializeAsString();
          }

          py::bytes step_stats_str = step_stats.SerializeAsString();
          return std::make_tuple(op_perf_objs, run_time, step_stats_str);
        });

  using DurationType = machina::grappler::Costs::Duration::rep;
  using MemoryUsage =
      std::tuple<std::string, int, size_t, DurationType, DurationType>;

  m.def(
      "TF_DeterminePeakMemoryUsage",
      [](machina::grappler::GrapplerItem* item,
         machina::grappler::Cluster* cluster)
          -> std::unordered_map<std::string,
                                std::tuple<int64_t, std::vector<MemoryUsage>>> {
        if (item == nullptr || cluster == nullptr) {
          tsl::MaybeRaiseRegisteredFromStatus(
              absl::Status(machina::errors::Internal(
                  "You need both a cluster and an item to determine peak "
                  "memory usage.")));
        }
        machina::grappler::GraphMemory memory(*item);

        if (cluster->DetailedStatsEnabled()) {
          tsl::MaybeRaiseRegisteredFromStatus(memory.InferDynamically(cluster));
        } else {
          tsl::MaybeRaiseRegisteredFromStatus(
              memory.InferStatically(cluster->GetDevices()));
        }

        std::unordered_map<std::string,
                           std::tuple<int64_t, std::vector<MemoryUsage>>>
            result;
        for (const auto& device : cluster->GetDevices()) {
          const machina::grappler::GraphMemory::MemoryUsage& usage =
              memory.GetPeakMemoryUsage(device.first);
          std::vector<MemoryUsage> per_device;
          for (size_t i = 0; i < usage.live_tensors.size(); ++i) {
            const auto& live_tensor = usage.live_tensors[i];
            per_device.push_back(std::make_tuple(
                live_tensor.node, live_tensor.output_id,
                live_tensor.memory_used, live_tensor.allocation_time.count(),
                live_tensor.deallocation_time.count()));
          }
          result[device.first] = std::make_tuple(usage.used_memory, per_device);
        }
        return result;
      });
}
