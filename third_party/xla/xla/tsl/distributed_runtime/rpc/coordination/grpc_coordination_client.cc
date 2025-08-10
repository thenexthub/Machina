/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/channel.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/generic/generic_stub.h"
#include "machina/xla/tsl/distributed_runtime/call_options.h"
#include "machina/xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "machina/xla/tsl/distributed_runtime/rpc/grpc_channel.h"
#include "machina/xla/tsl/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "machina/xla/tsl/distributed_runtime/rpc/grpc_state.h"
#include "machina/xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/protobuf.h"

namespace tsl {
namespace {
using machina::BarrierRequest;
using machina::BarrierResponse;
using machina::CancelBarrierRequest;
using machina::CancelBarrierResponse;
using machina::DeleteKeyValueRequest;
using machina::DeleteKeyValueResponse;
using machina::GetAliveTasksRequest;
using machina::GetAliveTasksResponse;
using machina::GetKeyValueDirRequest;
using machina::GetKeyValueDirResponse;
using machina::GetKeyValueRequest;
using machina::GetKeyValueResponse;
using machina::GetTaskStateRequest;
using machina::GetTaskStateResponse;
using machina::HeartbeatRequest;
using machina::HeartbeatResponse;
using machina::InsertKeyValueRequest;
using machina::InsertKeyValueResponse;
using machina::PollForErrorRequest;
using machina::PollForErrorResponse;
using machina::RegisterTaskRequest;
using machina::RegisterTaskResponse;
using machina::ReportErrorToServiceRequest;
using machina::ReportErrorToServiceResponse;
using machina::ReportErrorToTaskRequest;
using machina::ReportErrorToTaskResponse;
using machina::ResetTaskRequest;
using machina::ResetTaskResponse;
using machina::ShutdownTaskRequest;
using machina::ShutdownTaskResponse;
using machina::TryGetKeyValueRequest;
using machina::TryGetKeyValueResponse;
using machina::WaitForAllTasksRequest;
using machina::WaitForAllTasksResponse;
using machina::WatchJobStateRequest;
using machina::WatchJobStateResponse;

class GrpcCoordinationClientThread {
 public:
  GrpcCoordinationClientThread() {
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "coordination_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcCoordinationClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcCoordinationClientThread blocking for next tag";
          }
          VLOG(4) << "GrpcCoordinationClientThread exiting";
        }));
  }

  ~GrpcCoordinationClientThread() {
    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class GrpcCoordinationClient : public CoordinationClient {
 public:
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         ::grpc::CompletionQueue* cq, const std::string& target)
      : stub_(channel), cq_(cq), target_(target) {}
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         const std::string& target)
      : stub_(channel), target_(target) {
    client_thread_ = std::make_unique<GrpcCoordinationClientThread>();
    cq_ = client_thread_->completion_queue();
  }
  ~GrpcCoordinationClient() override = default;

  void RegisterTaskAsync(CallOptions* call_opts,
                         const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/RegisterTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/false,
        &target_);
  }

  void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                            WaitForAllTasksResponse* response,
                            StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/WaitForAllTasks",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ShutdownTaskAsync(CallOptions* call_opts,
                         const ShutdownTaskRequest* request,
                         ShutdownTaskResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/ShutdownTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ResetTaskAsync(const ResetTaskRequest* request,
                      ResetTaskResponse* response,
                      StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/ResetTask", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void HeartbeatAsync(CallOptions* call_opts, const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
    // Different from other RPCs which do not retry by default, the Heartbeat
    // RPC should retry automatically to tolerate transient network issues.
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/Heartbeat", *request,
        response, std::move(done), call_opts, /*threadpool=*/nullptr,
        /*max_retries=*/3,
        /*fail_fast=*/true, &target_);
  }

  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/ReportErrorToTask",
        *request, response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ReportErrorToServiceAsync(const ReportErrorToServiceRequest* request,
                                 ReportErrorToServiceResponse* response,
                                 StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/ReportErrorToService",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetTaskStateAsync(const GetTaskStateRequest* request,
                         GetTaskStateResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/GetTaskState", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void WatchJobStateAsync(CallOptions* call_opts,
                          const WatchJobStateRequest* request,
                          WatchJobStateResponse* response,
                          StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/WatchJobState", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/InsertKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueAsync(CallOptions* call_opts,
                        const GetKeyValueRequest* request,
                        GetKeyValueResponse* response,
                        StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/GetKeyValue", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                           TryGetKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/TryGetKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                           GetKeyValueDirResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/GetKeyValueDir", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                           DeleteKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/DeleteKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void BarrierAsync(CallOptions* call_opts, const BarrierRequest* request,
                    BarrierResponse* response, StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/Barrier", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void CancelBarrierAsync(const CancelBarrierRequest* request,
                          CancelBarrierResponse* response,
                          StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/CancelBarrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetAliveTasksAsync(const GetAliveTasksRequest* request,
                          GetAliveTasksResponse* response,
                          StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/GetAliveTasks", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void PollForErrorAsync(CallOptions* call_opts,
                         const PollForErrorRequest* request,
                         PollForErrorResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/machina.CoordinationService/PollForError", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  const std::string target_;
  std::unique_ptr<GrpcCoordinationClientThread> client_thread_;
};

class GrpcCoordinationClientCache : public CoordinationClientCache {
 public:
  explicit GrpcCoordinationClientCache(
      std::shared_ptr<GrpcChannelCache> channel_cache)
      : next_round_robin_assignment_(0),
        channel_cache_(channel_cache),
        threads_(4) {}

  ~GrpcCoordinationClientCache() override = default;

  CoordinationClient* GetClient(const std::string& target) override {
    absl::MutexLock l(&clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (channel == nullptr) {
        VLOG(2) << "Coordination client for target " << target << " not found.";
      }
      int assigned_index = AssignClientToThread(target);
      auto coord_client = std::make_unique<GrpcCoordinationClient>(
          channel, threads_[assigned_index].completion_queue(), target);
      it = clients_.emplace(target, std::move(coord_client)).first;
    }
    return it->second.get();
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const std::string& target) override {
    SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
    if (channel == nullptr) {
      VLOG(2) << "Coordination client for target " << target << " not found.";
    }
    return std::make_unique<GrpcCoordinationClient>(channel, target);
  }

 private:
  absl::Mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      ABSL_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ ABSL_GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const std::string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    absl::MutexLock l(&assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  std::shared_ptr<GrpcChannelCache> channel_cache_;
  mutable absl::Mutex clients_mu_;
  std::unordered_map<std::string, std::unique_ptr<CoordinationClient>> clients_
      ABSL_GUARDED_BY(clients_mu_);
  std::vector<GrpcCoordinationClientThread> threads_;
};

}  // namespace

CoordinationClientCache* NewGrpcCoordinationClientCache(
    std::shared_ptr<GrpcChannelCache> channel_cache) {
  return new GrpcCoordinationClientCache(channel_cache);
}

CoordinationClient* NewGrpcCoordinationClient(
    std::shared_ptr<::grpc::Channel> channel) {
  return new GrpcCoordinationClient(channel, /*target=*/"coordination_service");
}

}  // namespace tsl
