/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/core/tfrt/common/metrics.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "machina/xla/tsl/lib/monitoring/sampler.h"

namespace machina {
namespace tfrt_metrics {

tsl::monitoring::SamplerCell* GetTfrtGraphExecutorLatencySampler(
    const std::string& model_name, int64_t model_version,
    const std::string& graph_name) {
  static auto* cell = tsl::monitoring::Sampler<3>::New(
      {"/tfrt/graph_executor/latency",
       "Tracks the latency of GraphExecutor (in microseconds) of a graph.",
       "model_name", "model_version", "graph_name"},
      tsl::monitoring::Buckets::Exponential(10, 1.5, 33));
  return cell->GetCell(model_name, absl::StrCat(model_version), graph_name);
}

tsl::monitoring::SamplerCell* GetTfrtDeviceExecutionLatency(
    const std::string& model_name, int64_t model_version) {
  static auto* cell = tsl::monitoring::Sampler<2>::New(
      {"/tfrt/device_execution/latency",
       "Tracks the latency of device execution (in microseconds).",
       "model_name", "model_version"},
      tsl::monitoring::Buckets::Exponential(10, 1.5, 33));
  return cell->GetCell(model_name, absl::StrCat(model_version));
}

}  // namespace tfrt_metrics
}  // namespace machina
