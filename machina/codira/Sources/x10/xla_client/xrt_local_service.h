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

#ifndef X10_XLA_CLIENT_XRT_LOCAL_SERVICE_H_
#define X10_XLA_CLIENT_XRT_LOCAL_SERVICE_H_

#include <memory>
#include <string>

#include "machina/compiler/xla/xla_client/debug_macros.h"
#include "machina/compiler/xla/types.h"
#include "machina/core/distributed_runtime/server_lib.h"

namespace xla {

// A TF server running on a local interface.
class XrtLocalService {
 public:
  // The cluster_spec has format:
  //   CLUSTER_SPEC = JOB,...
  //   JOB          = NAME|ADDRESS_LIST
  //   NAME         = The name of the job
  //   ADDRESS_LIST = HOST:PORT;...
  //   HOST         = Hostname or IP address
  //   PORT         = Port number
  //
  // The job_name must match one of the job names in the cluster_spec, and
  // represents this job.
  // The task_index must be within the range of the ADDRESS_LIST of the current
  // job in the cluster_spec.
  XrtLocalService(const std::string& cluster_spec, const std::string& job_name,
                  int task_index);

  // Starts the service.
  void Start();

 private:
  std::unique_ptr<machina::ServerInterface> server_;
};

}  // namespace xla

#endif  // X10_XLA_CLIENT_XRT_LOCAL_SERVICE_H_
