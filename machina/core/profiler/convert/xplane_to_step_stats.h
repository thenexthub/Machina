/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_STATS_H_
#define MACHINA_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_STATS_H_

#include "machina/core/framework/step_stats.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace machina {
namespace profiler {

// Converts XSpace collected by profiling a GPU device to StepStats.
void ConvertGpuXSpaceToStepStats(const XSpace& xspace, StepStats* step_stats);

}  // namespace profiler
}  // namespace machina

#endif  // MACHINA_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_STATS_H_
