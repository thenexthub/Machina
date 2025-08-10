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

#ifndef MACHINA_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
#define MACHINA_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_

#include "machina/xla/tsl/framework/device_id.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Class that maintains a map from TfDeviceId to PlatformDeviceId, and manages
// the translation between them.
class GpuIdManager {
 public:
  // Adds a mapping from tf_device_id to platform_device_id.
  static absl::Status InsertTfPlatformDeviceIdPair(
      tsl::TfDeviceId tf_device_id, tsl::PlatformDeviceId platform_device_id);

  // Gets the platform_device_id associated with tf_device_id. Returns OK if
  // found.
  static absl::Status TfToPlatformDeviceId(
      tsl::TfDeviceId tf_device_id, tsl::PlatformDeviceId* platform_device_id);

  // Clears the map. Used in unit tests only.
  static void TestOnlyReset();
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
