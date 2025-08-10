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

#ifndef MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_REGISTRY_H_
#define MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "machina/core/common_runtime/cost_measurement.h"

namespace machina {

// CostMeasurementRegistry allows to
// - register a CostMeasurement type to the global map
// - create an instance of registered CostMeasurement.
class CostMeasurementRegistry {
 public:
  // Creates an instance of registered CostMeasurement by name. If the named
  // CostMeasurement is not registered yet, returns nullptr. Any returned
  // std::unique_ptr<CostMeasurement> should not be moved.
  // TODO(b/185852990): create a non-moveable wrapper class for the returned
  // unique_ptr<CostMeasurement>.
  static std::unique_ptr<CostMeasurement> CreateByNameOrNull(
      const std::string& name, const CostMeasurement::Context& context);

  using Creator = std::function<std::unique_ptr<CostMeasurement>(
      const CostMeasurement::Context&)>;

  // Registers a CostMeasurement type to the global map. Registering different
  // types of CostMeasurement with the same name is prohibited.
  static void RegisterCostMeasurement(absl::string_view name, Creator creator);
};

// Registers a CostMeasurement type to the global map. Registering different
// types of CostMeasurement with the same name is prohibited.
class CostMeasurementRegistrar {
 public:
  explicit CostMeasurementRegistrar(absl::string_view name,
                                    CostMeasurementRegistry::Creator creator) {
    CostMeasurementRegistry::RegisterCostMeasurement(name, std::move(creator));
  }
};

#define REGISTER_COST_MEASUREMENT(name, MyCostMeasurementClass)        \
  namespace {                                                          \
  static ::machina::CostMeasurementRegistrar                        \
      MyCostMeasurementClass##_registrar(                              \
          (name), [](const CostMeasurement::Context& context) {        \
            return std::make_unique<MyCostMeasurementClass>(context); \
          });                                                          \
  }  // namespace

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_REGISTRY_H_
