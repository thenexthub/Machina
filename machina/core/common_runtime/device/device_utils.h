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
#ifndef MACHINA_CORE_COMMON_RUNTIME_DEVICE_DEVICE_UTILS_H_
#define MACHINA_CORE_COMMON_RUNTIME_DEVICE_DEVICE_UTILS_H_

#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace device_utils {

// Validate device type. Device type must start with a capital letter and
// consist of capital letters and underscores. Reasoning behind this decision:
// * At the minimum we want to disallow '/' and ':' since
//   these characters are used in device spec, for e.g.
//   /job:foo/replica:12/device:GPU:1.
// * Underscores seem useful, for e.g. MACHINA_XLAGPU uses underscores.
// * Allowing lowercase might get confusing. For example, say someone
//   registers a new type called "Gpu". It might be confusing for users that
//   "Gpu" is not the same device type as "GPU".
//   Note that lowercase "cpu" and "gpu" are currently supported only for
//   legacy reasons:
//   https://cs.opensource.google/machina/machina/+/master:machina/python/framework/device_spec.py;l=46;drc=d3a378f9665d8eee827c74cb9ecbee81e4c288dd
absl::Status ValidateDeviceType(absl::string_view type);

}  // namespace device_utils
}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_DEVICE_DEVICE_UTILS_H_
