/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <array>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "machina/c/eager/c_api.h"
#include "machina/c/eager/parallel_device/parallel_device_lib.h"
#include "machina/c/eager/parallel_device/parallel_device_testlib.h"
#include "machina/c/tf_status.h"
#include "machina/core/distributed_runtime/master_env.h"
#include "machina/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/cluster.pb.h"
#include "machina/core/protobuf/machina_server.pb.h"

machina::ServerDef GetServerDef(const std::string& job_name, int num_tasks) {
  machina::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  machina::ClusterDef* cluster_def = server_def.mutable_cluster();
  machina::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = machina::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, machina::strings::StrCat("localhost", ":", port)});
    LOG(INFO) << "Picked test port: " << port << " for job: " << job_name
              << ", task: " << i;
  }
  return server_def;
}

namespace machina {
namespace parallel_device {

TEST(PARALLEL_DEVICE, TestRemoteBasic) {
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  machina::ServerDef server_def = GetServerDef("worker", 3);

  // This server def has the task index set to 0.
  std::string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<machina::GrpcServer> worker_server1;
  ASSERT_TRUE(machina::GrpcServer::Create(
                  server_def, machina::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<machina::GrpcServer> worker_server2;
  ASSERT_TRUE(machina::GrpcServer::Create(
                  server_def, machina::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TFE_ContextSetServerDef(context.get(), 0, serialized.data(),
                          serialized.size(), status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  BasicTestsForTwoDevices(context.get(),
                          "/job:worker/replica:0/task:1/device:CPU:0",
                          "/job:worker/replica:0/task:2/device:CPU:0");

  worker_server1.release();
  worker_server2.release();
}

TEST(PARALLEL_DEVICE, TestAsyncCopyOff) {
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  machina::ServerDef server_def = GetServerDef("worker", 3);

  // This server def has the task index set to 0.
  std::string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<machina::GrpcServer> worker_server1;
  ASSERT_TRUE(machina::GrpcServer::Create(
                  server_def, machina::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<machina::GrpcServer> worker_server2;
  ASSERT_TRUE(machina::GrpcServer::Create(
                  server_def, machina::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TFE_ContextSetServerDef(context.get(), 0, serialized.data(),
                          serialized.size(), status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const char* first_device = "/job:worker/replica:0/task:1/device:CPU:0";
  const char* second_device = "/job:worker/replica:0/task:2/device:CPU:0";
  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{first_device, second_device};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr value_one(FloatTensorHandle(3., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr value_two(FloatTensorHandle(-2., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::array<TFE_TensorHandle*, 2> in_components{value_one.get(),
                                                 value_two.get()};
  TensorHandlePtr combined_value = CreatePerDeviceValues(
      context.get(), in_components, device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Loop to make synchronization failures more deterministic
  for (int i = 0; i < 100; ++i) {
    TensorHandlePtr multiply_result(
        Multiply(context.get(), combined_value.get(), combined_value.get(),
                 status.get()));
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TensorHandlePtr, 2> out_components;
    ExtractPerDeviceValues(context.get(), multiply_result.get(),
                           &out_components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(out_components[0].get(), 9.);
    ExpectScalarEq<float>(out_components[1].get(), 4.);
  }

  worker_server1.release();
  worker_server2.release();
}
}  // namespace parallel_device
}  // namespace machina
