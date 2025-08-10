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
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics_collector_average_min_max.h"

#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics.pb.h"

namespace machina {
namespace calibrator {

void CalibrationStatisticsCollectorAverageMinMax::ClearData() {
  average_min_max_statistics_.set_min_sum(0.0);
  average_min_max_statistics_.set_max_sum(0.0);
  average_min_max_statistics_.set_num_samples(0);
}

void CalibrationStatisticsCollectorAverageMinMax::Collect(
    const float min, const float max, absl::Span<const int64_t> histogram) {
  const float current_min_sum = average_min_max_statistics_.min_sum();
  const float current_max_sum = average_min_max_statistics_.max_sum();
  const int current_num_samples = average_min_max_statistics_.num_samples();

  average_min_max_statistics_.set_min_sum(current_min_sum + min);
  average_min_max_statistics_.set_max_sum(current_max_sum + max);
  average_min_max_statistics_.set_num_samples(current_num_samples + 1);
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollectorAverageMinMax::GetStatistics() const {
  if (average_min_max_statistics_.num_samples() == 0) return std::nullopt;

  CalibrationStatistics statistics;
  statistics.mutable_average_min_max_statistics()->CopyFrom(
      average_min_max_statistics_);

  return statistics;
}

}  // namespace calibrator
}  // namespace machina
