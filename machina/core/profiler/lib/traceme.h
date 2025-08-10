/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_CORE_PROFILER_LIB_TRACEME_H_
#define MACHINA_CORE_PROFILER_LIB_TRACEME_H_

#include "absl/base/macros.h"
#include "machina/core/profiler/lib/traceme_encode.h"  // IWYU pragma: export
#include "tsl/profiler/lib/traceme.h"

namespace machina {
namespace profiler {

using tsl::profiler::kInfo;                                          // NOLINT
using TraceMe ABSL_DEPRECATE_AND_INLINE() = tsl::profiler::TraceMe;  // NOLINT
using TraceMeLevel ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::TraceMeLevel;  // NOLINT

ABSL_DEPRECATE_AND_INLINE()
inline int GetTFTraceMeLevel(bool is_expensive) {
  return tsl::profiler::GetTFTraceMeLevel(is_expensive);
}

ABSL_DEPRECATE_AND_INLINE()
inline bool TfOpDetailsEnabled() { return tsl::profiler::TfOpDetailsEnabled(); }

}  // namespace profiler
}  // namespace machina

#endif  // MACHINA_CORE_PROFILER_LIB_TRACEME_H_
