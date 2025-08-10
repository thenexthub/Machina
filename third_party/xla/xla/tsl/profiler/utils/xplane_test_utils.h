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
#ifndef MACHINA_MACHINA_XLA_TSL_PROFILER_UTILS_XPLANE_TEST_UTILS_H_
#define MACHINA_MACHINA_XLA_TSL_PROFILER_UTILS_XPLANE_TEST_UTILS_H_

#include <initializer_list>
#include <utility>
#include <variant>

#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "machina/xla/tsl/platform/types.h"
#include "machina/xla/tsl/profiler/utils/xplane_builder.h"
#include "machina/xla/tsl/profiler/utils/xplane_schema.h"

namespace tsl {
namespace profiler {

using XStatValue = std::variant<int64_t, uint64, absl::string_view>;

XPlane* GetOrCreateHostXPlane(XSpace* space);

XPlane* GetOrCreateGpuXPlane(XSpace* space, int32_t device_ordinal);

XPlane* GetOrCreateTpuXPlane(XSpace* space, int32_t device_ordinal,
                             absl::string_view device_type,
                             double peak_tera_flops_per_second,
                             double peak_hbm_bw_gigabytes_per_second);

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats = {});

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats = {});

void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name,
                               int64_t offset_ps, int64_t duration_ps,
                               absl::string_view execution_mode,
                               int64_t tracing_count = -1);
}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_MACHINA_XLA_TSL_PROFILER_UTILS_XPLANE_TEST_UTILS_H_
