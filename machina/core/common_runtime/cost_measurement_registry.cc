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

#include "machina/core/common_runtime/cost_measurement_registry.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "machina/core/common_runtime/cost_measurement.h"
#include "machina/core/platform/logging.h"

namespace machina {
namespace {

using RegistrationMap =
    absl::flat_hash_map<std::string, CostMeasurementRegistry::Creator>;

RegistrationMap* GetRegistrationMap() {
  static RegistrationMap* registered_cost_measurements = new RegistrationMap;
  return registered_cost_measurements;
}

}  // namespace

std::unique_ptr<CostMeasurement> CostMeasurementRegistry::CreateByNameOrNull(
    const std::string& name, const CostMeasurement::Context& context) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) {
    LOG_FIRST_N(ERROR, 1) << "Cost type " << name << " is unregistered.";
    return nullptr;
  }
  return it->second(context);
}

void CostMeasurementRegistry::RegisterCostMeasurement(absl::string_view name,
                                                      Creator creator) {
  const auto it = GetRegistrationMap()->find(name);
  CHECK(it == GetRegistrationMap()->end())  // Crash OK
      << "CostMeasurement " << name << " is registered twice.";
  GetRegistrationMap()->emplace(name, std::move(creator));
}

}  // namespace machina
