/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#include "machina/compiler/xla/xla_client/xrt_local_service.h"

#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/cluster.pb.h"
#include "machina/core/protobuf/machina_server.pb.h"
#include "machina/core/public/session_options.h"

namespace xla {
namespace {

void FillServerDef(const std::string& cluster_spec, const std::string& job_name,
                   int task_index, machina::ServerDef* options) {
  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);

  size_t my_num_tasks = 0;
  machina::ClusterDef* cluster = options->mutable_cluster();
  for (auto& job_str : absl::StrSplit(cluster_spec, ',')) {
    machina::JobDef* job_def = cluster->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    std::vector<std::string> job_pieces = absl::StrSplit(job_str, '|');
    XLA_CHECK_EQ(2, job_pieces.size()) << job_str;
    const std::string& cjob_name = job_pieces[0];
    const std::string& spec = job_pieces[1];
    job_def->set_name(cjob_name);
    std::vector<std::string> host_ports = absl::StrSplit(spec, ';');
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    size_t num_tasks = host_ports.size();
    if (job_name == options->job_name()) {
      my_num_tasks = num_tasks;
    }
    LOG(INFO) << "Peer " << cjob_name << " " << num_tasks << " {"
              << absl::StrJoin(host_ports, ", ") << "}";
  }
  XLA_CHECK_NE(my_num_tasks, 0) << "Job '" << options->job_name()
                                << "' does not appear in the cluster spec";
  XLA_CHECK_LT(options->task_index(), my_num_tasks)
      << "Task index " << options->task_index() << " is invalid (job '"
      << options->job_name() << "' contains " << my_num_tasks << " tasks";
}

}  // namespace

XrtLocalService::XrtLocalService(const std::string& cluster_spec,
                                 const std::string& job_name, int task_index) {
  machina::ServerDef server_def;
  FillServerDef(cluster_spec, job_name, task_index, &server_def);
  (*server_def.mutable_default_session_config()
      ->mutable_device_count())["GPU"] = 0;
  TF_CHECK_OK(machina::NewServer(server_def, &server_));
}

void XrtLocalService::Start() { TF_CHECK_OK(server_->Start()); }

}  // namespace xla
