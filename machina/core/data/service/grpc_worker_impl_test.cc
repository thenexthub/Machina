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
#include "machina/core/data/service/grpc_worker_impl.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/strings/str_cat.h"
#include "machina/core/data/service/common.pb.h"
#include "machina/core/data/service/credentials_factory.h"
#include "machina/core/data/service/dispatcher.pb.h"
#include "machina/core/data/service/server_lib.h"
#include "machina/core/data/service/test_util.h"
#include "machina/core/data/service/worker.pb.h"
#include "machina/core/distributed_runtime/rpc/grpc_util.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/data_service.pb.h"
#include "machina/core/protobuf/service_config.pb.h"

namespace machina {
namespace data {
namespace {

using ::grpc::Channel;
using ::grpc::ChannelArguments;
using ::grpc::ChannelCredentials;
using ::grpc::ClientContext;

constexpr const char kHostAddress[] = "localhost";
constexpr const char kProtocol[] = "grpc";

class GrpcWorkerImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK(SetUpDispatcherServer());
    TF_ASSERT_OK(SetUpWorkerServer());
    TF_ASSERT_OK(SetUpWorkerClientStub());
  }

  absl::Status SetUpDispatcherServer() {
    experimental::DispatcherConfig config;
    config.set_protocol(kProtocol);
    TF_RETURN_IF_ERROR(NewDispatchServer(config, dispatcher_server_));
    return dispatcher_server_->Start();
  }

  absl::Status SetUpWorkerServer() {
    experimental::WorkerConfig config;
    config.set_protocol(kProtocol);
    config.set_dispatcher_address(GetDispatcherAddress());
    config.set_worker_address(absl::StrCat(kHostAddress, ":%port%"));
    TF_RETURN_IF_ERROR(NewWorkerServer(config, worker_server_));
    return worker_server_->Start();
  }

  absl::Status SetUpWorkerClientStub() {
    std::shared_ptr<ChannelCredentials> credentials;
    TF_RETURN_IF_ERROR(
        CredentialsFactory::CreateClientCredentials(kProtocol, &credentials));
    ChannelArguments args;
    args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    std::shared_ptr<Channel> channel =
        ::grpc::CreateCustomChannel(GetWorkerAddress(), credentials, args);
    worker_client_stub_ = WorkerService::NewStub(channel);
    return absl::OkStatus();
  }

  std::string GetDispatcherAddress() const {
    return absl::StrCat(kHostAddress, ":", dispatcher_server_->BoundPort());
  }

  std::string GetWorkerAddress() const {
    return absl::StrCat(kHostAddress, ":", worker_server_->BoundPort());
  }

  std::unique_ptr<DispatchGrpcDataServer> dispatcher_server_;
  std::unique_ptr<WorkerGrpcDataServer> worker_server_;
  std::unique_ptr<WorkerService::Stub> worker_client_stub_;
};

TEST_F(GrpcWorkerImplTest, GetWorkerTasks) {
  ClientContext ctx;
  GetWorkerTasksRequest req;
  GetWorkerTasksResponse resp;
  TF_ASSERT_OK(
      FromGrpcStatus(worker_client_stub_->GetWorkerTasks(&ctx, req, &resp)));
  EXPECT_EQ(resp.tasks_size(), 0);
}

}  // namespace
}  // namespace data
}  // namespace machina
