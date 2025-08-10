/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef X10_XLA_CLIENT_TYPES_H_
#define X10_XLA_CLIENT_TYPES_H_

#include <cmath>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/types/optional.h"
#include "machina/compiler/xla/types.h"

namespace xla {

using hash_t = absl::uint128;

struct Percentile {
  enum class UnitOfMeaure {
    kNumber,
    kTime,
    kBytes,
  };
  struct Point {
    double percentile = 0.0;
    double value = 0.0;
  };

  UnitOfMeaure unit_of_measure = UnitOfMeaure::kNumber;
  uint64 start_nstime = 0;
  uint64 end_nstime = 0;
  double min_value = NAN;
  double max_value = NAN;
  double mean = NAN;
  double stddev = NAN;
  size_t num_samples = 0;
  size_t total_samples = 0;
  double accumulator = NAN;
  std::vector<Point> points;
};

struct Metric {
  absl::optional<Percentile> percentile;
  absl::optional<int64> int64_value;
};

}  // namespace xla

#endif  // X10_XLA_CLIENT_TYPES_H_
