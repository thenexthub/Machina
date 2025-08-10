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

#ifndef MACHINA_XLATSL_PROFILER_UTILS_DEVICE_UTILS_H_
#define MACHINA_XLATSL_PROFILER_UTILS_DEVICE_UTILS_H_

#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

enum class DeviceType {
  kUnknown,
  kCpu,
  kTpu,
  kGpu,
};

// Gets DeviceType from XPlane.
DeviceType GetDeviceType(const machina::profiler::XPlane& plane);
// Gets DeviceType from XPlane name.
DeviceType GetDeviceType(absl::string_view plane_name);

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_DEVICE_UTILS_H_
