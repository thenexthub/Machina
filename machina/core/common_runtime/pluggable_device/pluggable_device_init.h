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

#ifndef MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_INIT_H_
#define MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_INIT_H_

#include <string>

#include "absl/status/status.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/types.h"

namespace stream_executor {
class Platform;
}  // namespace stream_executor

namespace machina {

// Initializes the PluggableDevice platform and returns OK if the
// PluggableDevice platform could be initialized.
absl::Status ValidatePluggableDeviceMachineManager(const string& platform_name);

// Returns the PluggableDevice machine manager singleton, creating it and
// initializing the PluggableDevices on the machine if needed the first time it
// is called.  Must only be called when there is a valid PluggableDevice
// environment in the process (e.g., ValidatePluggableDeviceMachineManager()
// returns OK).
stream_executor::Platform* PluggableDeviceMachineManager(
    const string& platform_name);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_INIT_H_
