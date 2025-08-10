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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TESTLIB_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TESTLIB_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/subprocess.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session_options.h"

namespace machina {

class Device;

namespace test {

struct TestJob {
  std::string name;
  int num_tasks;
  int num_replicas = 1;
};

struct TestClusterConfig {
  std::string binary_path;
  SessionOptions options;
  std::vector<TestJob> jobs;

  TestClusterConfig& Options(const SessionOptions& options) {
    this->options = options;
    return *this;
  }
  TestClusterConfig& Jobs(const std::vector<TestJob>& jobs) {
    this->jobs = jobs;
    return *this;
  }
};

// Provides a handle to a set of TensorFlow servers (masters and
// workers) for testing purposes.
//
// This class currently runs the servers in separate processes; the
// lifetime of this object is coterminous with the lifetimes of those
// processes.
class TestCluster {
 public:
  // Creates a new test cluster based on the given `options` (which
  // configure the number of devices of each type) and a count of
  // processes `n`. On success, the test cluster is stored in
  // *out_cluster, and this function returns OK. Otherwise an error is
  // returned.
  static absl::Status MakeTestCluster(
      const TestClusterConfig& config,
      std::unique_ptr<TestCluster>* out_cluster);
  ~TestCluster();

  // Returns a vector of string "<hostname>:<port>" pairs that may be
  // used as targets to construct a GrpcSession.
  const std::vector<string>& targets(std::string job_name = "localhost") {
    return targets_.at(job_name);
  }

  // Returns a vector of devices available in this test cluster.
  const std::vector<DeviceAttributes>& devices() const { return devices_; }

 private:
  TestCluster() = default;

  std::vector<std::unique_ptr<SubProcess>> subprocesses_;
  absl::flat_hash_map<std::string, std::vector<std::string>> targets_;
  std::vector<DeviceAttributes> devices_;

  TestCluster(const TestCluster&) = delete;
  void operator=(const TestCluster&) = delete;
};

}  // end namespace test
}  // end namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TESTLIB_H_
