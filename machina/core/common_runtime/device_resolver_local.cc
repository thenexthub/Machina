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

#include "absl/status/status.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/platform/errors.h"

namespace machina {

absl::Status DeviceResolverLocal::GetDeviceAttributes(
    const string& device, DeviceAttributes* attributes) {
  Device* dev;
  // LookupDevice returns InvalidArgument if the device is not found.
  absl::Status s = dev_mgr_->LookupDevice(device, &dev);
  if (absl::IsInvalidArgument(s)) {
    return errors::NotFound(device, " not found");
  } else if (!s.ok()) {
    return s;
  }
  *attributes = dev->attributes();
  return absl::OkStatus();
}

absl::Status DeviceResolverLocal::GetAllDeviceAttributes(
    const string& task, std::vector<DeviceAttributes>* attributes) {
  return errors::Internal(
      "GetTaskCached is not supposed to be called in local collectives");
}

absl::Status DeviceResolverLocal::UpdateDeviceAttributes(
    const std::vector<DeviceAttributes>& attributes) {
  return errors::Internal(
      "UpdateDeviceAttributes shouldn't be called with local collectives");
}

}  // namespace machina
