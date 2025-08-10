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

#ifndef MACHINA_CORE_PLATFORM_TRACING_H_
#define MACHINA_CORE_PLATFORM_TRACING_H_

// Tracing interface

#include <array>

#include "machina/core/platform/macros.h"
#include "machina/core/platform/platform.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/tracing.h"

namespace machina {
namespace tracing {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::tracing::EventCategory;
using tsl::tracing::EventCollector;
using tsl::tracing::GetArgForName;
using tsl::tracing::GetEventCategoryName;
using tsl::tracing::GetEventCollector;
using tsl::tracing::GetLogDir;
using tsl::tracing::GetNumEventCategories;
using tsl::tracing::GetUniqueArg;
using tsl::tracing::RecordEvent;
using tsl::tracing::ScopedRegion;
using tsl::tracing::SetEventCollector;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tracing
}  // namespace machina

#if defined(PLATFORM_GOOGLE)
#include "machina/xla/tsl/platform/google/tracing_impl.h"
#else
#include "machina/xla/tsl/platform/default/tracing_impl.h"
#endif

#endif  // MACHINA_CORE_PLATFORM_TRACING_H_
