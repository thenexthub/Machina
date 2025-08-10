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

#include "machina/core/data/service/grpc_dispatcher_impl.h"

#include "grpcpp/server_context.h"
#include "machina/core/data/service/export.pb.h"
#include "machina/core/distributed_runtime/rpc/grpc_util.h"
#include "machina/core/protobuf/service_config.pb.h"

namespace machina {
namespace data {

using ::grpc::ServerBuilder;
using ::grpc::ServerContext;

GrpcDispatcherImpl::GrpcDispatcherImpl(
    const experimental::DispatcherConfig& config, ServerBuilder& server_builder)
    : impl_(config) {
  server_builder.RegisterService(this);
  VLOG(1) << "Registered data service dispatcher";
}

absl::Status GrpcDispatcherImpl::Start() { return impl_.Start(); }

void GrpcDispatcherImpl::Stop() { impl_.Stop(); }

size_t GrpcDispatcherImpl::NumActiveIterations() {
  return impl_.NumActiveIterations();
}

DispatcherStateExport GrpcDispatcherImpl::ExportState() const {
  return impl_.ExportState();
}

#define HANDLER(method)                                                   \
  grpc::Status GrpcDispatcherImpl::method(ServerContext* context,         \
                                          const method##Request* request, \
                                          method##Response* response) {   \
    return ToGrpcStatus(impl_.method(request, response));                 \
  }
HANDLER(WorkerHeartbeat);
HANDLER(WorkerUpdate);
HANDLER(GetDatasetDef);
HANDLER(GetSplit);
HANDLER(GetVersion);
HANDLER(GetOrRegisterDataset);
HANDLER(ReleaseIterationClient);
HANDLER(MaybeRemoveTask);
HANDLER(GetOrCreateJob);
HANDLER(GetOrCreateIteration);
HANDLER(ClientHeartbeat);
HANDLER(GetWorkers);
HANDLER(GetDataServiceMetadata);
HANDLER(GetDataServiceConfig);
HANDLER(Snapshot);
HANDLER(GetSnapshotSplit);
HANDLER(GetSnapshotStreams);
HANDLER(DisableCompressionAtRuntime);
#undef HANDLER

}  // namespace data
}  // namespace machina
