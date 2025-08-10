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

#ifndef MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_
#define MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/common_runtime/device/device_id.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/device_factory.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session_options.h"

namespace machina {
class PluggableDeviceFactory : public DeviceFactory {
 public:
  PluggableDeviceFactory(const string& device_type,
                         const string& platform_name);
  absl::Status ListPhysicalDevices(std::vector<string>* devices) override;
  absl::Status CreateDevices(
      const SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) override;
  absl::Status GetDeviceDetails(
      int device_index, std::unordered_map<string, string>* details) override;

 private:
  // Populates *device_localities with the DeviceLocality descriptor for
  // every TfDeviceId.
  absl::Status GetDeviceLocalities(
      int num_tf_devices, std::vector<DeviceLocality>* device_localities);
  // Create a PluggableDevice associated with 'tf_device_id', allocates
  // (strictly) 'memory_limit' bytes of PluggableDevice memory to it, and adds
  // it to the 'devices' vector.
  absl::Status CreatePluggableDevice(
      const SessionOptions& options, const std::string& name_prefix,
      TfDeviceId tf_device_id, int64_t memory_limit,
      const DeviceLocality& dev_locality,
      std::vector<std::unique_ptr<Device>>* devices);

  const string device_type_;
  const string platform_name_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_
