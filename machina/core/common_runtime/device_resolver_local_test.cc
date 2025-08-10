/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "machina/core/common_runtime/device_resolver_local.h"

#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"

namespace machina {
namespace {

#define NUM_DEVS 3

class DeviceResolverLocalTest : public ::testing::Test {
 protected:
  DeviceResolverLocalTest() {
    SessionOptions options;
    string task_name = "/job:localhost/replica:0/task:0";
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
};

TEST_F(DeviceResolverLocalTest, GetDeviceAttributesKnown) {
  DeviceAttributes attributes;
  TF_EXPECT_OK(drl_->GetDeviceAttributes(
      "/job:localhost/replica:0/task:0/device:CPU:1", &attributes));
  EXPECT_EQ(attributes.name(), "/job:localhost/replica:0/task:0/device:CPU:1");
}

TEST_F(DeviceResolverLocalTest, GetDeviceAttributesUnknown) {
  DeviceAttributes attributes;
  EXPECT_TRUE(absl::IsNotFound(drl_->GetDeviceAttributes(
      "/job:localhost/replica:0/task:0/device:CPU:9", &attributes)));
}

TEST_F(DeviceResolverLocalTest, GetAllDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  EXPECT_TRUE(
      absl::IsInternal(drl_->GetAllDeviceAttributes(/*task*/ "", &attributes)));
}

TEST_F(DeviceResolverLocalTest, UpdateDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  EXPECT_TRUE(absl::IsInternal(drl_->UpdateDeviceAttributes(attributes)));
}

}  // namespace
}  // namespace machina
