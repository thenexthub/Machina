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

#ifndef MACHINA_CORE_PROFILER_UTILS_TRACE_UTILS_H_
#define MACHINA_CORE_PROFILER_UTILS_TRACE_UTILS_H_

#include "machina/xla/tsl/profiler/utils/trace_utils.h"

namespace machina {
namespace profiler {

using tsl::profiler::IsDerivedThreadId;            // NOLINT
using tsl::profiler::kFirstDeviceId;               // NOLINT
using tsl::profiler::kHostThreadsDeviceId;         // NOLINT
using tsl::profiler::kLastDeviceId;                // NOLINT
using tsl::profiler::kThreadIdDerivedMax;          // NOLINT
using tsl::profiler::kThreadIdDerivedMin;          // NOLINT
using tsl::profiler::kThreadIdHloModule;           // NOLINT
using tsl::profiler::kThreadIdHloOp;               // NOLINT
using tsl::profiler::kThreadIdHostOffloadOpEnd;    // NOLINT
using tsl::profiler::kThreadIdHostOffloadOpStart;  // NOLINT
using tsl::profiler::kThreadIdKernelLaunch;        // NOLINT
using tsl::profiler::kThreadIdOverhead;            // NOLINT
using tsl::profiler::kThreadIdSource;              // NOLINT
using tsl::profiler::kThreadIdStepInfo;            // NOLINT
using tsl::profiler::kThreadIdTfNameScope;         // NOLINT
using tsl::profiler::kThreadIdTfOp;                // NOLINT

}  // namespace profiler
}  // namespace machina

#endif  // MACHINA_CORE_PROFILER_UTILS_TRACE_UTILS_H_
