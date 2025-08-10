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

#include "machina/core/common_runtime/pluggable_device/pluggable_device_init.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/xla/stream_executor/platform_manager.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"

namespace machina {

absl::Status ValidatePluggableDeviceMachineManager(
    const string& platform_name) {
  return se::PlatformManager::PlatformWithName(platform_name).status();
}

se::Platform* PluggableDeviceMachineManager(const string& platform_name) {
  auto result = se::PlatformManager::PlatformWithName(platform_name);
  if (!result.ok()) {
    LOG(FATAL) << "Could not find platform with name "  // Crash OK
               << platform_name;
    return nullptr;
  }
  return result.value();
}

}  // namespace machina
