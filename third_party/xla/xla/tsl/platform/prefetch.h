/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_XLATSL_PLATFORM_PREFETCH_H_
#define MACHINA_XLATSL_PLATFORM_PREFETCH_H_

#include "absl/base/prefetch.h"

namespace tsl {
namespace port {

// Prefetching support.
// Deprecated. Prefer to call absl::Prefetch* directly.

enum PrefetchHint {
  PREFETCH_HINT_T0 = 3,  // Temporal locality
  PREFETCH_HINT_NTA = 0  // No temporal locality
};

template <PrefetchHint hint>
void prefetch(const void* x) {
  absl::PrefetchToLocalCache(x);
}

template <>
inline void prefetch<PREFETCH_HINT_NTA>(const void* x) {
  absl::PrefetchToLocalCacheNta(x);
}

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_PREFETCH_H_
