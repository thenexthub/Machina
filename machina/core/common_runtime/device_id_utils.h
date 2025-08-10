
/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_DEVICE_ID_UTILS_H_
#define MACHINA_CORE_COMMON_RUNTIME_DEVICE_ID_UTILS_H_

#include "machina/xla/stream_executor/platform.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/xla/tsl/framework/device_id.h"
#include "machina/xla/tsl/framework/device_id_manager.h"

namespace machina {

// Utility method for getting the associated executor given a TfDeviceId.
class DeviceIdUtil {
 public:
  static absl::StatusOr<stream_executor::StreamExecutor*> ExecutorForTfDeviceId(
      const tsl::DeviceType& type, stream_executor::Platform* device_manager,
      tsl::TfDeviceId tf_device_id) {
    tsl::PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(tsl::DeviceIdManager::TfToPlatformDeviceId(
        type, tf_device_id, &platform_device_id));
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_DEVICE_ID_UTILS_H_
