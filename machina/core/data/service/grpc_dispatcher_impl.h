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

#ifndef MACHINA_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_
#define MACHINA_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_

#include "grpcpp/server_builder.h"
#include "machina/core/data/service/dispatcher.grpc.pb.h"
#include "machina/core/data/service/dispatcher_impl.h"
#include "machina/core/data/service/export.pb.h"
#include "machina/core/protobuf/service_config.pb.h"

namespace machina {
namespace data {

// This class is a wrapper that handles communication for gRPC.
class GrpcDispatcherImpl : public DispatcherService::Service {
 public:
  // Constructs a GrpcDispatcherImpl with the given config, and registers it
  // with `server_builder`.
  explicit GrpcDispatcherImpl(const experimental::DispatcherConfig& config,
                              ::grpc::ServerBuilder& server_builder);
  ~GrpcDispatcherImpl() override { Stop(); }

  absl::Status Start();
  void Stop();

  size_t NumActiveIterations();

  DispatcherStateExport ExportState() const;

#define HANDLER(method)                                 \
  ::grpc::Status method(::grpc::ServerContext* context, \
                        const method##Request* request, \
                        method##Response* response) override;
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

 private:
  DataServiceDispatcherImpl impl_;

  GrpcDispatcherImpl(const GrpcDispatcherImpl&) = delete;
  void operator=(const GrpcDispatcherImpl&) = delete;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_
