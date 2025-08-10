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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/support/byte_buffer.h"

#include "machina/core/distributed_runtime/rpc/grpc_util.h"
#include "machina/core/distributed_runtime/tensor_coding.h"
#include "machina/core/protobuf/worker.pb.h"

namespace grpc {

// Support parsing/unparsing of machina::TensorResponse.
// Wire-format is identical to RecvTensorResponse.
// This is specializing an existing template, so it's okay to do this in a
// namespace that we don't own.
template <>
class SerializationTraits<machina::TensorResponse> {
 public:
  static Status Serialize(const machina::TensorResponse& msg, ByteBuffer* bp,
                          bool* own_buffer) {
    LOG(FATAL) << "TODO(sanjay,jeff): Implement";
    return Status();
  }
  static Status Deserialize(ByteBuffer* buffer,
                            machina::TensorResponse* msg) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }
    Status result = Status::OK;
    if (result.ok()) {
      ::machina::GrpcByteSource source(buffer);
      auto s = msg->ParseFrom(&source);
      if (!s.ok()) {
        result = Status(StatusCode::INTERNAL,
                        ::machina::strings::StrCat(
                            "TensorResponse parse error", s.message()));
      }
    }
    buffer->Clear();
    return result;
  }
};

}  // namespace grpc

namespace machina {

// Names of worker methods.
enum class GrpcWorkerMethod {
  kGetStatus,
  kCreateWorkerSession,
  kDeleteWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kRecvBuf,
  kLogging,
  kTracing,
  kCompleteGroup,
  kCompleteInstance,
  kGetStepSequence,
  kMarkRecvFinished,
};

static const int kGrpcNumWorkerMethods =
    static_cast<int>(GrpcWorkerMethod::kMarkRecvFinished) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc {

// Implementation of `machina.WorkerService`, based on the
// definition in "//machina/core/protobuf/worker_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class WorkerService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace grpc

}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
