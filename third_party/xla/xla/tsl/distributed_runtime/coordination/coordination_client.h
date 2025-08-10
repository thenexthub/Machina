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

#ifndef MACHINA_MACHINA_XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_
#define MACHINA_MACHINA_XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_

#include <memory>
#include <string>

#include "machina/xla/tsl/distributed_runtime/call_options.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/protobuf/coordination_service.pb.h"

namespace tsl {
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

// Base class of client interface for communicating with coordination service.
// Can be implemented by a variety of transports such as gRPC.
class CoordinationClient {
 public:
  virtual ~CoordinationClient() = default;

  virtual void RegisterTaskAsync(CallOptions* call_opts,
                                 const RegisterTaskRequest* request,
                                 RegisterTaskResponse* response,
                                 StatusCallback done) = 0;

  virtual void HeartbeatAsync(CallOptions* call_opts,
                              const HeartbeatRequest* request,
                              HeartbeatResponse* response,
                              StatusCallback done) = 0;

  virtual void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                                    WaitForAllTasksResponse* response,
                                    StatusCallback done) = 0;

  virtual void ShutdownTaskAsync(CallOptions* call_opts,
                                 const ShutdownTaskRequest* request,
                                 ShutdownTaskResponse* response,
                                 StatusCallback done) = 0;

  virtual void ResetTaskAsync(const ResetTaskRequest* request,
                              ResetTaskResponse* response,
                              StatusCallback done) = 0;

  virtual void ReportErrorToTaskAsync(CallOptions* call_opts,
                                      const ReportErrorToTaskRequest* request,
                                      ReportErrorToTaskResponse* response,
                                      StatusCallback done) = 0;

  virtual void ReportErrorToServiceAsync(
      const ReportErrorToServiceRequest* request,
      ReportErrorToServiceResponse* response, StatusCallback done) = 0;

  virtual void GetTaskStateAsync(const GetTaskStateRequest* request,
                                 GetTaskStateResponse* response,
                                 StatusCallback done) = 0;

  virtual void WatchJobStateAsync(CallOptions* call_opts,
                                  const WatchJobStateRequest* request,
                                  WatchJobStateResponse* response,
                                  StatusCallback done) = 0;

  virtual void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                                   InsertKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void GetKeyValueAsync(CallOptions* call_opts,
                                const GetKeyValueRequest* request,
                                GetKeyValueResponse* response,
                                StatusCallback done) = 0;

  virtual void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                                   TryGetKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                                   GetKeyValueDirResponse* response,
                                   StatusCallback done) = 0;

  virtual void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                                   DeleteKeyValueResponse* response,
                                   StatusCallback done) = 0;

  virtual void BarrierAsync(CallOptions* call_opts,
                            const BarrierRequest* request,
                            BarrierResponse* response, StatusCallback done) = 0;

  virtual void CancelBarrierAsync(const CancelBarrierRequest* request,
                                  CancelBarrierResponse* response,
                                  StatusCallback done) = 0;

  virtual void GetAliveTasksAsync(const GetAliveTasksRequest* request,
                                  GetAliveTasksResponse* response,
                                  StatusCallback done) = 0;

  virtual void PollForErrorAsync(CallOptions* call_opts,
                                 const PollForErrorRequest* request,
                                 PollForErrorResponse* response,
                                 StatusCallback done) = 0;
};

// Simple wrapper class that can be used to retrieve CoordinationClients.
class CoordinationClientCache {
 public:
  virtual ~CoordinationClientCache() = default;

  // If the `target` names a remote task, returns a pointer of the
  // CoordinationClient object wrapping that channel to the remote task.
  virtual CoordinationClient* GetClient(const std::string& target) = 0;

  // If the `target` names a remote task, returns an owned pointer of the
  // CoordinationClient object wrapping that channel to the remote task.
  virtual std::unique_ptr<CoordinationClient> GetOwnedClient(
      const std::string& target) = 0;
};

}  // namespace tsl

#endif  // MACHINA_MACHINA_XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_CLIENT_H_
