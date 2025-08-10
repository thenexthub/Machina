/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "machina/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "machina/core/framework/device_factory.h"

namespace machina {

class NextPluggableDeviceFactory : public DeviceFactory {
 public:
  explicit NextPluggableDeviceFactory(
      const std::string& device_type,
      const std::string& compilation_device_name)
      : api_(TfnpdApi()),
        device_type_(device_type),
        compilation_device_name_(compilation_device_name) {}

  absl::Status ListPhysicalDevices(std::vector<string>* devices) override;

  absl::Status CreateDevices(
      const SessionOptions& session_options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) override;

  const std::string& compilation_device_name() const {
    return compilation_device_name_;
  }

 private:
  const TFNPD_Api* api_;
  const std::string device_type_;
  const std::string compilation_device_name_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_
