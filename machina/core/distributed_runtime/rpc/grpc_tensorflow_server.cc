/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include <iostream>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "machina/core/distributed_runtime/server_lib.h"

#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/protobuf/cluster.pb.h"
#include "machina/core/protobuf/machina_server.pb.h"
#include "machina/core/public/session_options.h"
#include "machina/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker).
//
// TODO(mrry): Replace with a py_binary that uses `tf.GrpcServer()`.
namespace machina {
namespace {

absl::Status FillServerDef(const string& cluster_spec, const string& job_name,
                           int task_index, ServerDef* options) {
  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);

  size_t my_num_tasks = 0;

  ClusterDef* const cluster = options->mutable_cluster();

  for (const string& job_str : str_util::Split(cluster_spec, ',')) {
    JobDef* const job_def = cluster->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    const string& job_name = job_pieces[0];
    job_def->set_name(job_name);
    // Does a bit more validation of the tasks_per_replica.
    const absl::string_view spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    size_t num_tasks = host_ports.size();
    if (job_name == options->job_name()) {
      my_num_tasks = host_ports.size();
    }
    LOG(INFO) << "Peer " << job_name << " " << num_tasks << " {"
              << absl::StrJoin(host_ports, ", ") << "}";
  }
  if (my_num_tasks == 0) {
    return errors::InvalidArgument("Job name \"", options->job_name(),
                                   "\" does not appear in the cluster spec");
  }
  if (options->task_index() >= my_num_tasks) {
    return errors::InvalidArgument("Task index ", options->task_index(),
                                   " is invalid (job \"", options->job_name(),
                                   "\" contains ", my_num_tasks, " tasks");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace machina

void Usage(char* const argv_0) {
  std::cerr << "Usage: " << argv_0
            << " --cluster_spec=SPEC --job_name=NAME --task_id=ID" << std::endl;
  std::cerr << "Where:" << std::endl;
  std::cerr << "    SPEC is <JOB>(,<JOB>)*" << std::endl;
  std::cerr << "    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*" << std::endl;
  std::cerr << "    NAME is a valid job name ([a-z][0-9a-z]*)" << std::endl;
  std::cerr << "    HOST is a hostname or IP address" << std::endl;
  std::cerr << "    PORT is a port number" << std::endl;
}

int main(int argc, char* argv[]) {
  machina::string cluster_spec;
  machina::string job_name;
  int task_index = 0;
  std::vector<machina::Flag> flag_list = {
      machina::Flag("cluster_spec", &cluster_spec, "cluster spec"),
      machina::Flag("job_name", &job_name, "job name"),
      machina::Flag("task_id", &task_index, "task id"),
  };
  machina::string usage = machina::Flags::Usage(argv[0], flag_list);
  const bool parse_result = machina::Flags::Parse(&argc, argv, flag_list);
  machina::port::InitMain(argv[0], &argc, &argv);
  if (!parse_result || argc != 1) {
    std::cerr << usage << std::endl;
    Usage(argv[0]);
    return -1;
  }
  machina::ServerDef server_def;
  absl::Status s = machina::FillServerDef(cluster_spec, job_name, task_index,
                                             &server_def);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s.message() << std::endl;
    Usage(argv[0]);
    return -1;
  }
  std::unique_ptr<machina::ServerInterface> server;
  TF_QCHECK_OK(machina::NewServer(server_def, &server));
  TF_QCHECK_OK(server->Start());
  TF_QCHECK_OK(server->Join());
}
