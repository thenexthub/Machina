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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_

#include <memory>

#include "absl/flags/flag.h"
#include "machina/c/kernels.h"
#include "machina/c/tf_status_helper.h"
#include "machina/core/common_runtime/next_pluggable_device/c_plugin_coordination_service_agent.h"
#include "machina/core/common_runtime/next_pluggable_device/direct_plugin_coordination_service_agent.h"
#include "machina/core/common_runtime/next_pluggable_device/flags.h"
#include "machina/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent.h"

namespace machina {

inline std::unique_ptr<PluginCoordinationServiceAgent>
CreatePluginCoordinationServiceAgent(void* agent) {
  if (!absl::GetFlag(FLAGS_next_pluggable_device_use_c_api)) {
    return std::make_unique<DirectPluginCoordinationServiceAgent>(agent);
  } else {
    return std::make_unique<CPluginCoordinationServiceAgent>(agent);
  }
}

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
