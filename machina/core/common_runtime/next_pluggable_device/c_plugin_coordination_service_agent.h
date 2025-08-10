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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_COORDINATION_SERVICE_AGENT_H_
#define MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_COORDINATION_SERVICE_AGENT_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "machina/c/experimental/next_pluggable_device/c_api.h"
#include "machina/c/kernels_experimental.h"
#include "machina/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"

namespace machina {

class CPluginCoordinationServiceAgent : public PluginCoordinationServiceAgent {
 public:
  explicit CPluginCoordinationServiceAgent(void* agent)
      : agent_(reinterpret_cast<TF_CoordinationServiceAgent*>(agent)) {}

  bool IsInitialized() const override {
    if (agent_ == nullptr) return false;
    return TF_CoordinationServiceIsInitialized(agent_);
  }

  absl::Status InsertKeyValue(std::string_view key,
                              std::string_view value) override;

  absl::StatusOr<std::string> GetKeyValue(std::string_view key) override;
  absl::StatusOr<std::string> GetKeyValue(std::string_view key,
                                          absl::Duration timeout) override;
  absl::StatusOr<std::string> TryGetKeyValue(std::string_view key) override;

  absl::Status DeleteKeyValue(std::string_view key) override;

 private:
  TF_CoordinationServiceAgent* agent_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_COORDINATION_SERVICE_AGENT_H_
