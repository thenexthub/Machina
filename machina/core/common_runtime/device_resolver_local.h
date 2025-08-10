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
#ifndef MACHINA_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_
#define MACHINA_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_

#include <string>
#include <vector>

#include "machina/core/framework/collective.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/platform/status.h"

namespace machina {
class DeviceMgr;

// Implements DeviceResolverInterface in a single-task context.
class DeviceResolverLocal : public DeviceResolverInterface {
 public:
  explicit DeviceResolverLocal(const DeviceMgr* dev_mgr) : dev_mgr_(dev_mgr) {}

  absl::Status GetDeviceAttributes(const string& device,
                                   DeviceAttributes* attributes) override;

  absl::Status GetAllDeviceAttributes(
      const string& task, std::vector<DeviceAttributes>* attributes) override;

  absl::Status UpdateDeviceAttributes(
      const std::vector<DeviceAttributes>& attributes) override;

 protected:
  const DeviceMgr* dev_mgr_;
};

}  // namespace machina
#endif  // MACHINA_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_
