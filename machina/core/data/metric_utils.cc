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
#include "machina/core/data/metric_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "machina/core/data/utils.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/metrics.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/mutex.h"

namespace machina {
namespace data {
namespace {

// Safely subtracts `y` from `x` avoiding underflow.
uint64_t safe_sub(uint64_t x, uint64_t y) { return x >= y ? x - y : 0; }

}  // namespace

IteratorMetricsCollector::IteratorMetricsCollector(
    const std::string& device_type, const Env& env)
    : device_type_(device_type), env_(env) {}

absl::Time IteratorMetricsCollector::RecordStart() {
  const uint64_t start_time_us = env_.NowMicros();
  if (!ShouldCollectMetrics()) {
    return absl::FromUnixMicros(start_time_us);
  }

  mutex_lock l(mu_);
  if (end_time_us_ == 0) {
    // We initialize `end_time_us_` to the start time of the first request to
    // make it possible to use the delta between `end_time_us_` and subsequent
    // `GetNext()` end time to incrementally collect the duration of the
    // iterator's lifetime.
    end_time_us_ = start_time_us;
  }
  uint64_t gap_time_us = 0;
  if (num_active_calls_ == 0) {
    first_start_time_us_ = start_time_us;
    gap_time_us = safe_sub(start_time_us, end_time_us_);
  }
  metrics::RecordTFDataIteratorGap(gap_time_us);
  num_active_calls_++;
  return absl::FromUnixMicros(start_time_us);
}

void IteratorMetricsCollector::RecordStop(absl::Time start_time,
                                          const std::vector<Tensor>& output) {
  if (!ShouldCollectMetrics()) {
    return;
  }

  const uint64_t end_time_us = env_.NowMicros();
  const int64_t latency_micros =
      safe_sub(end_time_us, absl::ToUnixMicros(start_time));
  AddLatencySample(latency_micros);
  IncrementThroughput(GetTotalBytes(output));
  mutex_lock l(mu_);
  metrics::RecordTFDataIteratorLifetime(safe_sub(end_time_us, end_time_us_));
  end_time_us_ = std::max(end_time_us_, end_time_us);
  num_active_calls_--;
  if (num_active_calls_ == 0) {
    metrics::RecordTFDataIteratorBusy(
        safe_sub(end_time_us_, first_start_time_us_));
  }
}

bool IteratorMetricsCollector::ShouldCollectMetrics() const {
  return device_type_ == DEVICE_CPU;
}

}  // namespace data
}  // namespace machina
