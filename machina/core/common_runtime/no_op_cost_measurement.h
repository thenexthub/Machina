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

#ifndef MACHINA_CORE_COMMON_RUNTIME_NO_OP_COST_MEASUREMENT_H_
#define MACHINA_CORE_COMMON_RUNTIME_NO_OP_COST_MEASUREMENT_H_

#include "absl/strings/string_view.h"
#include "machina/core/common_runtime/cost_measurement.h"
#include "machina/core/common_runtime/cost_measurement_registry.h"

namespace machina {

// This class does not do the real cost measurement. It will always return zero
// Duration as the total cost. It's created to allow callers to skip collecting
// costs.
class NoOpCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  // Always returns zero Duration as the total cost.
  absl::Duration GetTotalCost() override;
  absl::string_view GetCostType() const override;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_NO_OP_COST_MEASUREMENT_H_
