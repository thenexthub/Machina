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

// This file defines a recorder for op cost measurement.

#ifndef MACHINA_CORE_TFRT_FALLBACK_COST_RECORDER_H_
#define MACHINA_CORE_TFRT_FALLBACK_COST_RECORDER_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/thread_annotations.h"

namespace machina {
namespace tfrt_stub {

// Thread-safe.
// Maintains the execution durations by `op_key`. Note that `op_key` is only
// unique within a model.
class CostRecorder {
 public:
  // Records an execution duration for the op keyed by `op_key`.
  void RecordCost(int64_t op_key, uint64_t execution_time);

  // Returns the normalized average execution duration of the op keyed by
  // `op_key`. If there is no record for `op_key`, returns the uint32_t::max to
  // avoid stream merging. Note that we don't use uint64_t::max because
  // otherwise adding op costs would cause overflow.
  uint64_t GetCost(int64_t op_key) const;

  // Writes the op cost map (in format of `OpCostMapProto`) to a file specified
  // by the env var name `MesuredCostPathEnvVarName()`.
  // TODO(b/263837451): Fix the op_key unstableness during serialization.
  absl::Status WriteToFile() const;

  size_t size() const;

  static const char* MesuredCostPathEnvVarName() {
    return "TF_TFRT_MEASURED_COST_PATH";
  }

 private:
  mutable machina::mutex op_cost_map_mutex_;
  // Map op key to {sum of op execution duration, #occurences of the op}.
  absl::flat_hash_map<int64_t, std::pair<uint64_t, uint64_t>> op_cost_map_
      TF_GUARDED_BY(op_cost_map_mutex_);
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_FALLBACK_COST_RECORDER_H_
