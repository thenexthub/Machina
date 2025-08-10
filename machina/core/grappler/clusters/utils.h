/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_CLUSTERS_UTILS_H_
#define MACHINA_CORE_GRAPPLER_CLUSTERS_UTILS_H_

#include "machina/core/common_runtime/gpu/gpu_id.h"
#include "machina/core/protobuf/device_properties.pb.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace grappler {

// Returns the DeviceProperties of the CPU on which grappler is running.
DeviceProperties GetLocalCPUInfo();

// Returns the DeviceProperties for the specified GPU attached to the server on
// which grappler is running.
DeviceProperties GetLocalGPUInfo(PlatformDeviceId platform_device_id);

// Returns the DeviceProperties of the specified device
DeviceProperties GetDeviceInfo(const DeviceNameUtils::ParsedName& device);

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_CLUSTERS_UTILS_H_
