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

#include "machina/core/tpu/kernels/tpu_compilation_metrics.h"

#if defined(LIBTPU_ON_GCE)

#include <cstdint>

#include "absl/strings/string_view.h"

namespace machina {
namespace tpu {

// TODO(b/295556102): remove this once `TpuCompilationCache` migration to OSS is
// completed.

void TpuCompilationMetrics::IncrementCacheLookupCount(
    bool is_cache_hit, absl::string_view session_name) {
  // A placeholder for tracking metrics.
}

/* static */
void TpuCompilationMetrics::SetCacheEntryCount(int64_t count) {
  // A placeholder for tracking metrics.
}

/* static */
void TpuCompilationMetrics::IncrementCompilationCount(
    absl::string_view session_name) {
  // A placeholder for tracking metrics.
}

}  // namespace tpu
}  // namespace machina

#endif  // LIBTPU_ON_GCE
