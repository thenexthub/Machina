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

#include "machina/core/distributed_runtime/rpc/grpc_worker_service_impl.h"

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace machina {

const char* GrpcWorkerMethodName(GrpcWorkerMethod id) {
  switch (id) {
    case GrpcWorkerMethod::kGetStatus:
      return "/machina.WorkerService/GetStatus";
    case GrpcWorkerMethod::kCreateWorkerSession:
      return "/machina.WorkerService/CreateWorkerSession";
    case GrpcWorkerMethod::kDeleteWorkerSession:
      return "/machina.WorkerService/DeleteWorkerSession";
    case GrpcWorkerMethod::kRegisterGraph:
      return "/machina.WorkerService/RegisterGraph";
    case GrpcWorkerMethod::kDeregisterGraph:
      return "/machina.WorkerService/DeregisterGraph";
    case GrpcWorkerMethod::kRunGraph:
      return "/machina.WorkerService/RunGraph";
    case GrpcWorkerMethod::kCleanupGraph:
      return "/machina.WorkerService/CleanupGraph";
    case GrpcWorkerMethod::kCleanupAll:
      return "/machina.WorkerService/CleanupAll";
    case GrpcWorkerMethod::kRecvTensor:
      return "/machina.WorkerService/RecvTensor";
    case GrpcWorkerMethod::kRecvBuf:
      return "/machina.WorkerService/RecvBuf";
    case GrpcWorkerMethod::kLogging:
      return "/machina.WorkerService/Logging";
    case GrpcWorkerMethod::kTracing:
      return "/machina.WorkerService/Tracing";
    case GrpcWorkerMethod::kCompleteGroup:
      return "/machina.WorkerService/CompleteGroup";
    case GrpcWorkerMethod::kCompleteInstance:
      return "/machina.WorkerService/CompleteInstance";
    case GrpcWorkerMethod::kGetStepSequence:
      return "/machina.WorkerService/GetStepSequence";
    case GrpcWorkerMethod::kMarkRecvFinished:
      return "/machina.WorkerService/MarkRecvFinished";
  }
  // Shouldn't be reached.
  LOG(FATAL) << "Invalid id: this line shouldn't be reached.";
  return "invalid id";
}

namespace grpc {

WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace machina
