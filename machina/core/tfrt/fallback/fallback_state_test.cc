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
#include "machina/core/tfrt/fallback/fallback_state.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include "absl/base/nullability.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/const_op.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/platform/status_matchers.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/error_codes.pb.h"

namespace machina {
namespace {

using ::machina::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(FallbackStateTest, CreateWithCpuDeviceVector) {
  machina::SessionOptions session_options;
  machina::FunctionDefLibrary fdef_lib;

  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  std::variant<std::vector<std::unique_ptr<Device>>,
               DynamicDeviceMgr* absl_nonnull>
      device_variant = std::move(devices);

  auto fallback_state = std::make_unique<tfrt_stub::FallbackState>(
      session_options, std::move(device_variant), fdef_lib);

  const auto& device_manager = fallback_state->device_manager();
  EXPECT_GT(device_manager.NumDevices(), 0);
  EXPECT_EQ(device_manager.NumDeviceType("CPU"), 1);
}

TEST(FallbackStateTest, CreateWithDynamicDeviceMgr) {
  machina::SessionOptions session_options;
  machina::FunctionDefLibrary fdef_lib;

  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  auto static_device_mgr =
      std::make_unique<DynamicDeviceMgr>(std::move(devices));

  DynamicDeviceMgr* absl_nonnull device_mgr_ptr(static_device_mgr.get());

  auto fallback_state = std::make_unique<tfrt_stub::FallbackState>(
      session_options, device_mgr_ptr, fdef_lib);
  const auto& device_manager = fallback_state->device_manager();

  EXPECT_GT(device_manager.NumDevices(), 0);
  EXPECT_EQ(device_manager.NumDeviceType("CPU"), 1);
}

TEST(FallbackStateTest, CreateRendezvous) {
  // Given a FallbackState, when a function is launched by function library
  // runtime without an explicit rendezvous, it should be able to create one
  // from the rendezvous factory.
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*function_name=*/"dummy_fn",
      /*arg_def=*/{},
      /*return values=*/{},
      /*attr def=*/{},
      /*node_def=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          tfrt_stub::FallbackState::Create({}, flib));

  const ProcessFunctionLibraryRuntime& pflr =
      fallback_state->process_function_library_runtime();
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:localhost/replica:0/task:0";
  opts.remote_execution = true;

  auto status = pflr.RunSync(opts, pflr.GetHandle("dummy_fn"), {}, nullptr);

  EXPECT_THAT(status, Not(StatusIs(error::FAILED_PRECONDITION,
                                   HasSubstr("rendezvous"))));
}

TEST(FallbackStateTest, CreateGraphExecutionState) {
  machina::SessionOptions session_options;
  machina::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tfrt_stub::FallbackState::CreateWithCpuDevice(session_options, fdef_lib));

  GraphDef graphdef;
  {
    auto scope = machina::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto graph_execution_state,
                          fallback_state->CreateGraphExecutionState(
                              std::move(graphdef), /*run_placer=*/true,
                              /*enable_tf2xla_mlir_bridge=*/false));
}

TEST(FallbackStateTest, CreateWithMockGpuDevice) {
  machina::SessionOptions session_options;
  machina::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          tfrt_stub::FallbackState::CreateWithMockGpuDevice(
                              session_options, fdef_lib));
  const auto& device_manager = fallback_state->device_manager();
  EXPECT_GT(device_manager.NumDeviceType("GPU"), 0);
}

}  // namespace
}  // namespace machina
