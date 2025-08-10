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

#ifndef MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_H_
#define MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_H_

#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace machina {

// An interface for cost measurement.
class CostMeasurement {
 public:
  // Context of the CostMeasurement.
  struct Context {
    // Whether this CostMeasurement is running within a per-query context (e.g.
    // rpc handler) or not (e.g. batching).
    bool is_per_query = false;
  };

  explicit CostMeasurement(const Context& context) {}

  virtual ~CostMeasurement() {}

  virtual absl::Duration GetTotalCost() = 0;

  virtual absl::string_view GetCostType() const = 0;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_COST_MEASUREMENT_H_
