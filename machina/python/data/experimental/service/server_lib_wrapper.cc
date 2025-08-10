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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Python.h"
#include "absl/status/status.h"
#include "pybind11/chrono.h"  // from @pybind11
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "machina/core/data/service/common.pb.h"
#include "machina/core/data/service/dispatcher_client.h"
#include "machina/core/data/service/grpc_util.h"
#include "machina/core/data/service/server_lib.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/data_service.pb.h"
#include "machina/core/protobuf/service_config.pb.h"
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_server_lib, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<machina::data::DispatchGrpcDataServer>(m,
                                                       "DispatchGrpcDataServer")
      .def("start", &machina::data::DispatchGrpcDataServer::Start)
      .def("stop", &machina::data::DispatchGrpcDataServer::Stop)
      .def("join", &machina::data::DispatchGrpcDataServer::Join,
           py::call_guard<py::gil_scoped_release>())
      .def("bound_port", &machina::data::DispatchGrpcDataServer::BoundPort)
      .def("num_workers",
           [](machina::data::DispatchGrpcDataServer* server) -> int {
             int num_workers;
             absl::Status status = server->NumWorkers(&num_workers);
             machina::MaybeRaiseFromStatus(status);
             return num_workers;
           })
      .def("snapshot_streams",
           [](machina::data::DispatchGrpcDataServer* server,
              const std::string& path)
               -> std::vector<machina::data::SnapshotStreamInfoWrapper> {
             std::vector<machina::data::SnapshotStreamInfoWrapper> streams;
             absl::Status status = server->SnapshotStreams(path, &streams);
             machina::MaybeRaiseFromStatus(status);
             return streams;
           });

  py::class_<machina::data::WorkerGrpcDataServer>(m, "WorkerGrpcDataServer")
      .def("start", &machina::data::WorkerGrpcDataServer::Start)
      .def("stop", &machina::data::WorkerGrpcDataServer::Stop)
      .def("join", &machina::data::WorkerGrpcDataServer::Join,
           py::call_guard<py::gil_scoped_release>())
      .def("bound_port", &machina::data::WorkerGrpcDataServer::BoundPort)
      .def("num_tasks",
           [](machina::data::WorkerGrpcDataServer* server) -> int {
             int num_tasks;
             absl::Status status = server->NumTasks(&num_tasks);
             machina::MaybeRaiseFromStatus(status);
             return num_tasks;
           })
      .def("snapshot_task_progresses",
           [](machina::data::WorkerGrpcDataServer* server)
               -> std::vector<machina::data::SnapshotTaskProgressWrapper> {
             std::vector<machina::data::SnapshotTaskProgressWrapper>
                 snapshot_task_progresses;
             absl::Status status =
                 server->SnapshotTaskProgresses(&snapshot_task_progresses);
             machina::MaybeRaiseFromStatus(status);
             return snapshot_task_progresses;
           });

  m.def(
      "TF_DATA_NewDispatchServer",
      [](std::string serialized_dispatcher_config)
          -> std::unique_ptr<machina::data::DispatchGrpcDataServer> {
        machina::data::experimental::DispatcherConfig config;
        if (!config.ParseFromString(serialized_dispatcher_config)) {
          machina::MaybeRaiseFromStatus(machina::errors::InvalidArgument(
              "Failed to deserialize dispatcher config."));
        }
        std::unique_ptr<machina::data::DispatchGrpcDataServer> server;
        absl::Status status =
            machina::data::NewDispatchServer(config, server);
        machina::MaybeRaiseFromStatus(status);
        return server;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_DATA_NewWorkerServer",
      [](std::string serialized_worker_config)
          -> std::unique_ptr<machina::data::WorkerGrpcDataServer> {
        machina::data::experimental::WorkerConfig config;
        if (!config.ParseFromString(serialized_worker_config)) {
          machina::MaybeRaiseFromStatus(machina::errors::InvalidArgument(
              "Failed to deserialize worker config."));
        }
        std::unique_ptr<machina::data::WorkerGrpcDataServer> server;
        absl::Status status = machina::data::NewWorkerServer(config, server);
        machina::MaybeRaiseFromStatus(status);
        return server;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_DATA_GetDataServiceMetadataByID",
      [](std::string dataset_id, const std::string& address,
         const std::string& protocol) -> machina::data::DataServiceMetadata {
        machina::data::DataServiceMetadata metadata;
        machina::data::DataServiceDispatcherClient client(address, protocol);
        int64_t deadline_micros = machina::kint64max;
        absl::Status status;
        Py_BEGIN_ALLOW_THREADS;
        status = machina::data::grpc_util::Retry(
            [&]() {
              return client.GetDataServiceMetadata(dataset_id, metadata);
            },
            /*description=*/
            machina::strings::StrCat(
                "Get data service metadata for dataset ", dataset_id,
                " from dispatcher at ", address),
            deadline_micros);
        Py_END_ALLOW_THREADS;
        machina::MaybeRaiseFromStatus(status);
        return metadata;
      },
      py::return_value_policy::reference);

  py::class_<machina::data::SnapshotTaskProgressWrapper>
      snapshot_task_progress_wrapper(m, "SnapshotTaskProgressWrapper");
  snapshot_task_progress_wrapper.def(py::init<>())
      .def_property_readonly(
          "snapshot_task_base_path",
          [](const machina::data::SnapshotTaskProgressWrapper&
                 snapshot_task_progress_wrapper) -> py::bytes {
            return snapshot_task_progress_wrapper.snapshot_task_base_path;
          })
      .def_property_readonly(
          "snapshot_task_stream_index",
          [](const machina::data::SnapshotTaskProgressWrapper&
                 snapshot_task_progress_wrapper) -> int {
            return snapshot_task_progress_wrapper.snapshot_task_stream_index;
          })
      .def_property_readonly(
          "completed",
          [](const machina::data::SnapshotTaskProgressWrapper&
                 snapshot_task_progress_wrapper) -> bool {
            return snapshot_task_progress_wrapper.completed;
          });
  py::class_<machina::data::SnapshotStreamInfoWrapper>
      snapshot_stream_info_wrapper(m, "SnapshotStreamInfoWrapper");
  snapshot_stream_info_wrapper.def(py::init<>())
      .def_property_readonly(
          "index",
          [](const machina::data::SnapshotStreamInfoWrapper&
                 snapshot_stream_info_wrapper) -> int {
            return snapshot_stream_info_wrapper.index;
          })
      .def_property_readonly(
          "state",
          [](const machina::data::SnapshotStreamInfoWrapper&
                 snapshot_stream_info_wrapper) -> int {
            return snapshot_stream_info_wrapper.state;
          });
};
