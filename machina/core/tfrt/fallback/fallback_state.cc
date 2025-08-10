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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/graph_execution_state.h"
#include "machina/core/common_runtime/rendezvous_mgr.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/rendezvous.h"
#include "machina/core/framework/types.h"
#include "machina/core/graph/types.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session_options.h"
#include "machina/core/public/version.h"
#include "machina/core/tpu/virtual_device.h"
#include "tsl/platform/refcount.h"

namespace machina {
namespace tfrt_stub {

namespace {

string DeviceName(absl::string_view name_prefix, absl::string_view device_type,
                  int32_t task_id, size_t device_id) {
  return strings::StrCat(absl::StripSuffix(name_prefix, "0"), task_id,
                         "/device:", device_type, ":", device_id);
}

DeviceAttributes BuildDeviceAttributes(absl::string_view name_prefix,
                                       const char *device_type, int32_t task_id,
                                       size_t device_id) {
  const DeviceAttributes attrs = Device::BuildDeviceAttributes(
      DeviceName(name_prefix, device_type, task_id, device_id),
      DeviceType(device_type), Bytes(16ULL << 30), DeviceLocality(),
      strings::StrCat("device: ", device_type, " device"));
  return attrs;
}

}  // namespace

absl::StatusOr<std::unique_ptr<FallbackState>> FallbackState::Create(
    const SessionOptions &session_options,
    const machina::FunctionDefLibrary &fdef_lib) {
  // Create devices.
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}

absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithCpuDevice(
    const SessionOptions &session_options,
    const machina::FunctionDefLibrary &fdef_lib) {
  // Create devices.
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}

absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithMockGpuDevice(
    const SessionOptions &session_options,
    const machina::FunctionDefLibrary &fdef_lib) {
  // Create devices.
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  auto device_attrs =
      BuildDeviceAttributes("/job:localhost/replica:0/task:0", "GPU", 0, 0);
  devices.push_back(
      std::make_unique<VirtualDevice>(session_options.env, device_attrs));

  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}

absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithDeviceMgr(
    const SessionOptions &session_options,
    const machina::FunctionDefLibrary &fdef_lib,
    DynamicDeviceMgr *absl_nonnull device_mgr) {
  return std::make_unique<FallbackState>(session_options, device_mgr, fdef_lib);
}

FallbackState::FallbackState(const SessionOptions &session_options,
                             std::variant<std::vector<std::unique_ptr<Device>>,
                                          DynamicDeviceMgr *absl_nonnull>
                                 device_mgr,
                             const machina::FunctionDefLibrary &fdef_lib)
    : session_options_(session_options),
      device_manager_(
          std::holds_alternative<std::vector<std::unique_ptr<Device>>>(
              device_mgr)
              ? std::move(
                    std::get<std::vector<std::unique_ptr<Device>>>(device_mgr))
              : std::vector<std::unique_ptr<Device>>()),
      device_manager_ptr_(
          std::holds_alternative<DynamicDeviceMgr *absl_nonnull>(device_mgr)
              ? std::get<DynamicDeviceMgr *absl_nonnull>(device_mgr)
              : &device_manager_),
      func_lib_def_(OpRegistry::Global(), fdef_lib),
      pflr_(device_manager_ptr_, session_options.env, &session_options.config,
            TF_GRAPH_DEF_VERSION, &func_lib_def_,
            session_options.config.graph_options().optimizer_options(),
            /*thread_pool=*/nullptr, /*parent=*/nullptr,
            /*session_metadata=*/nullptr,
            Rendezvous::Factory{[](const int64_t, const DeviceMgr *device_mgr,
                                   tsl::core::RefCountPtr<Rendezvous> *r) {
              *r = tsl::core::RefCountPtr<Rendezvous>(
                  new IntraProcessRendezvous(device_mgr));
              return absl::OkStatus();
            }}) {
  for (auto *d : device_manager_ptr_->ListDevices()) {
    device_set_.AddDevice(d);
  }

  // client_device is the device for feed and fetch tensors.
  device_set_.set_client_device(device_manager().HostCPU());
}

absl::StatusOr<std::unique_ptr<GraphExecutionState>>
FallbackState::CreateGraphExecutionState(GraphDef graph_def, bool run_placer,
                                         bool enable_tf2xla_mlir_bridge) const {
  // Create GraphExecutionState which contains the preprocessed graph including
  // device information. The following code is adapted from
  // http://cs?q=machina/core/common_runtime/direct_session.cc:427%20at_cl:352783230

  GraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &session_options_;
  options.session_handle = "tfrt_fallback_handle";
  options.run_placer = run_placer;
  options.enable_tf2xla_mlir_bridge = enable_tf2xla_mlir_bridge;

  std::unique_ptr<GraphExecutionState> execution_state;
  TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
      std::move(graph_def), options, &execution_state));
  return execution_state;
}

absl::Status FallbackState::AddFunctionDef(const FunctionDef &func_def) {
  return func_lib_def_.AddFunctionDef(func_def);
}

}  // namespace tfrt_stub
}  // namespace machina
