/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/xla/tsl/profiler/utils/device_utils.h"

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

DeviceType GetDeviceType(absl::string_view plane_name) {
  if (plane_name == kHostThreadsPlaneName) {
    return DeviceType::kCpu;
  } else if (absl::StartsWith(plane_name, kTpuPlanePrefix)) {
    return DeviceType::kTpu;
  } else if (absl::StartsWith(plane_name, kGpuPlanePrefix)) {
    return DeviceType::kGpu;
  } else {
    return DeviceType::kUnknown;
  }
}
DeviceType GetDeviceType(const machina::profiler::XPlane& plane) {
  return GetDeviceType(plane.name());
}
}  // namespace profiler
}  // namespace tsl
