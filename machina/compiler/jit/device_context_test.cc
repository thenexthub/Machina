/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "machina/compiler/jit/flags.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/tensor_testutil.h"

namespace machina {
namespace {

static bool Initialized = [] {
  auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
  rollout_config.enabled_for_xla_launch_ = true;
  rollout_config.enabled_for_compile_on_demand_ = true;

  machina::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

class DeviceContextTest : public ::testing::Test {
 public:
  void SetDevice(const string& device_type) {
    auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
    rollout_config.AllowForDeviceInXlaLaunch(DeviceType(device_type));
    rollout_config.AllowForDeviceInXlaCompileOnDemand(DeviceType(device_type));

    auto device_factory = DeviceFactory::GetFactory(device_type);
    SessionOptions options;
    std::vector<std::unique_ptr<Device>> devices;
    absl::Status s = device_factory->CreateDevices(
        options, "/job:worker/replica:0/task:0", &devices);
    device_ = std::move(devices[0]);

    machina::AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_allocator_ = device_->GetAllocator(host_alloc_attr);

    machina::AllocatorAttributes device_alloc_attr;
    device_alloc_attr.set_on_host(false);
    device_allocator_ = device_->GetAllocator(device_alloc_attr);

    machina::DeviceContext* device_context;
    auto status = device_->TryGetDeviceContext(&device_context);
    TF_EXPECT_OK(status);
    device_context_.reset(device_context);
  }

  std::unique_ptr<Device> device_;
  machina::core::RefCountPtr<DeviceContext> device_context_;
  machina::Allocator* host_allocator_;
  machina::Allocator* device_allocator_;
};

#if GOOGLE_CUDA || MACHINA_USE_ROCM
TEST_F(DeviceContextTest, TestXlaGpuRoundTripTransferWithDeviceApi) {
  SetDevice(DEVICE_MACHINA_MACHINA_XLA_GPU);
  machina::Tensor origin_cpu_tensor(host_allocator_, machina::DT_FLOAT,
                                       machina::TensorShape({2, 2}));
  machina::test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  machina::Tensor device_tensor(device_allocator_, machina::DT_FLOAT,
                                   machina::TensorShape({2, 2}));
  machina::Tensor dest_cpu_tensor(host_allocator_, machina::DT_FLOAT,
                                     machina::TensorShape({2, 2}));

  TF_ASSERT_OK(device_context_->CopyCPUTensorToDeviceSync(
      &origin_cpu_tensor, device_.get(), &device_tensor));
  TF_ASSERT_OK(device_context_->CopyDeviceTensorToCPUSync(
      &device_tensor, "", device_.get(), &dest_cpu_tensor));
  LOG(INFO) << "H2D - D2H roundtrip completes. tensor: "
            << dest_cpu_tensor.DebugString(4);

  machina::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
}
#endif

}  // namespace
}  // namespace machina
