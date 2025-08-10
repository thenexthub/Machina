/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_METRICS_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_METRICS_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace machina {
namespace tpu {

// Tracks Tpu compilation and cache metrics.
class TpuCompilationMetrics {
 public:
  // Increments the number of cache lookup count.
  static void IncrementCacheLookupCount(bool is_cache_hit,
                                        absl::string_view session_name);

  // Sets the total count of cache entries.
  static void SetCacheEntryCount(int64_t count);

  // Increments number of compilation.
  static void IncrementCompilationCount(absl::string_view session_name);
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_COMPILATION_METRICS_H_
