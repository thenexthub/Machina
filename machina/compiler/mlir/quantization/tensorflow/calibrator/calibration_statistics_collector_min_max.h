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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_MIN_MAX_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_MIN_MAX_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics.pb.h"
#include "machina/compiler/mlir/quantization/machina/calibrator/calibration_statistics_collector_base.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace machina {
namespace calibrator {

using ::stablehlo::quantization::CalibrationOptions;

// MinMax calibration calculates the global min and global max values.
// global min = min of given sample inputs
// global max = max of given sample inputs
class CalibrationStatisticsCollectorMinMax
    : public CalibrationStatisticsCollectorBase {
 public:
  explicit CalibrationStatisticsCollectorMinMax() { ClearData(); }

  void ClearData() override;

  void Collect(float min, float max,
               absl::Span<const int64_t> histogram) override;

  std::optional<CalibrationStatistics> GetStatistics() const override;

 private:
  CalibrationStatistics::MinMaxStatistics min_max_statistics_;
};

}  // namespace calibrator
}  // namespace machina
#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_MIN_MAX_H_
