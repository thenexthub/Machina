/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics_collector_min_max.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/types/span.h"
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics.pb.h"

namespace machina {
namespace calibrator {

void CalibrationStatisticsCollectorMinMax::ClearData() {
  // global_min will be updated by std::min(global_min, input_value) so
  // it is initialized with the value numeric_limits<float>::max().
  min_max_statistics_.set_global_min(std::numeric_limits<float>::max());

  // global_max will be updated by std::max(global_max, input_value) so it
  // is initialized with the value numeric_limits<float>::lowest().
  min_max_statistics_.set_global_max(std::numeric_limits<float>::lowest());
}

void CalibrationStatisticsCollectorMinMax::Collect(
    const float min, const float max, absl::Span<const int64_t> histogram) {
  min_max_statistics_.set_global_min(
      std::min(min_max_statistics_.global_min(), min));
  min_max_statistics_.set_global_max(
      std::max(min_max_statistics_.global_max(), max));
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollectorMinMax::GetStatistics() const {
  if (min_max_statistics_.global_min() == std::numeric_limits<float>::max())
    return std::nullopt;

  CalibrationStatistics statistics;
  statistics.mutable_min_max_statistics()->CopyFrom(min_max_statistics_);

  return statistics;
}

}  // namespace calibrator
}  // namespace machina
