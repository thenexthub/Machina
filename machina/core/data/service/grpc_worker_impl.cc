/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/core/data/service/grpc_worker_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "machina/core/data/service/export.pb.h"
#include "machina/core/data/service/worker_impl.h"
#include "machina/core/distributed_runtime/rpc/grpc_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/service_config.pb.h"

namespace machina {
namespace data {

using ::grpc::ServerBuilder;
using ::grpc::ServerContext;

GrpcWorkerImpl::GrpcWorkerImpl(const experimental::WorkerConfig& config,
                               ServerBuilder& server_builder)
    : impl_(std::make_shared<DataServiceWorkerImpl>(config)) {
  server_builder.RegisterService(this);
  VLOG(1) << "Registered data service worker";
}

absl::Status GrpcWorkerImpl::Start(
    const std::string& worker_address,
    const std::vector<DataTransferServerInfo>& transfer_servers) {
  worker_address_ = worker_address;
  TF_RETURN_IF_ERROR(impl_->Start(worker_address, transfer_servers));
  LocalWorkers::Add(worker_address, impl_);
  return absl::OkStatus();
}

void GrpcWorkerImpl::Stop() {
  LocalWorkers::Remove(worker_address_);
  impl_->Stop();
}

WorkerStateExport GrpcWorkerImpl::ExportState() const {
  return impl_->ExportState();
}

#define HANDLER(method)                                                 \
  ::grpc::Status GrpcWorkerImpl::method(ServerContext* context,         \
                                        const method##Request* request, \
                                        method##Response* response) {   \
    return ToGrpcStatus(impl_->method(request, response));              \
  }
HANDLER(ProcessTask);
HANDLER(GetElement);
HANDLER(GetWorkerTasks);
HANDLER(GetSnapshotTaskProgresses);
#undef HANDLER

}  // namespace data
}  // namespace machina
