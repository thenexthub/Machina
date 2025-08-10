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
#include "machina/compiler/mlir/machina/utils/fake_session.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "toolchain/Support/CommandLine.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/threadpool_device.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/resource_var.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/types.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "machina/core/public/session_options.h"

namespace mlir {
namespace TF {
namespace test_util {
namespace {
using ::machina::Status;
using ::machina::Tensor;

const char kDeviceNamePrefix[] = "/job:worker/replica:0/task:1";
const char kDeviceName[] = "/job:worker/replica:0/task:1/device:CPU:0";

// Struct holding options for FakeSession which are configuered through
// command line flags.
struct FakeSessionOptions {
  toolchain::cl::opt<bool> fail_to_fetch_local_device_manager{
      "fail-to-fetch-local-device-manager",
      toolchain::cl::desc("Fail to fetch local device manager."),
      toolchain::cl::init(false)};
};
FakeSessionOptions* kSessionOptions = []() { return new FakeSessionOptions; }();
}  // namespace

FakeSession::FakeSession() {
  // We don't initialize devices in constructor as it causes some
  // global initialization fiasco between tests and code in TF.
}

void FakeSession::Initialize() {
  if (initialized_) return;
  BuildDeviceManager();
  InitVariables();
  initialized_ = true;
}

void FakeSession::BuildDeviceManager() {
  auto device =
      machina::DeviceFactory::NewDevice("CPU", {}, kDeviceNamePrefix);
  device_mgr_ =
      std::make_unique<machina::StaticDeviceMgr>(std::move(device));
}

void FakeSession::InitVariables() {
  machina::Device* device = nullptr;
  auto status = device_mgr_->LookupDevice(kDeviceName, &device);
  if (status != absl::OkStatus()) return;
  auto container = device->resource_manager()->default_container();

  // Create 2 resources and initialize them with dummy values.
  TF_CHECK_OK(device->resource_manager()->Create(
      container, "var1", new machina::Var(machina::DataType::DT_FLOAT)));
  TF_CHECK_OK(device->resource_manager()->Create(
      container, "var2", new machina::Var(machina::DataType::DT_FLOAT)));
}

Status FakeSession::Create(const machina::GraphDef& graph) {
  return machina::errors::Unimplemented("not available");
}
Status FakeSession::Extend(const machina::GraphDef& graph) {
  return machina::errors::Unimplemented("not available");
}

Status FakeSession::Close() {
  return machina::errors::Unimplemented("not available");
}

Status FakeSession::ListDevices(
    std::vector<machina::DeviceAttributes>* response) {
  return machina::errors::Unimplemented("not available");
}

Status FakeSession::LocalDeviceManager(
    const machina::DeviceMgr** deviceMgrPtr) {
  Initialize();
  if (kSessionOptions->fail_to_fetch_local_device_manager)
    return Status(absl::StatusCode::kUnknown, "No Local Device Manager");
  *deviceMgrPtr = device_mgr_.get();
  return absl::OkStatus();
}

Status FakeSession::Run(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes,
    std::vector<Tensor>* outputs) {
  machina::RunMetadata run_metadata;
  return Run(machina::RunOptions(), inputs, output_names, target_nodes,
             outputs, &run_metadata);
}

Status FakeSession::Run(
    const machina::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    machina::RunMetadata* run_metadata) {
  return Run(run_options, inputs, output_names, target_nodes, outputs,
             run_metadata, machina::thread::ThreadPoolOptions());
}

Status FakeSession::Run(
    const machina::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    machina::RunMetadata* run_metadata,
    const machina::thread::ThreadPoolOptions& thread_pool_options) {
  Initialize();
  for (const std::string& output_name : output_names) {
    Tensor output;
    if (output_name == "dense/bias") {
      Tensor t = Tensor(machina::DT_FLOAT, machina::TensorShape({50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "dense/kernel") {
      Tensor t =
          Tensor(machina::DT_FLOAT, machina::TensorShape({100, 50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "var1") {
      Tensor t = Tensor(machina::DT_RESOURCE, machina::TensorShape({1}));
      t.scalar<machina::ResourceHandle>()().set_name("var1");
      t.scalar<machina::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var2") {
      Tensor t = Tensor(machina::DT_RESOURCE, machina::TensorShape({1}));
      t.scalar<machina::ResourceHandle>()().set_name("var2");
      t.scalar<machina::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var3") {
      Tensor t = Tensor(machina::DT_RESOURCE, machina::TensorShape({1}));
      t.scalar<machina::ResourceHandle>()().set_name("var3");
      t.scalar<machina::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "invalid_var") {
      Tensor t = Tensor(machina::DT_RESOURCE, machina::TensorShape({1}));
      t.scalar<machina::ResourceHandle>()().set_name("invalid_var");
      t.scalar<machina::ResourceHandle>()().set_device("invalid_device");

      outputs->push_back(t);
    } else if (absl::StartsWith(output_name, "var")) {
      return Status(absl::StatusCode::kNotFound,
                    "Can't find variable " + output_name + " in session");
    } else {
      // Create a scalar float tensor.
      Tensor t = Tensor(machina::DT_FLOAT, machina::TensorShape({}));
      t.flat<float>()(0) = 1.0f;
      outputs->push_back(t);
    }
  }
  return absl::OkStatus();
}

}  // namespace test_util
}  // namespace TF
}  // namespace mlir
