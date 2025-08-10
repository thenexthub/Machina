/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_LIB_MONITORING_SAMPLER_H_
#define MACHINA_CORE_LIB_MONITORING_SAMPLER_H_

#include "machina/xla/tsl/lib/monitoring/sampler.h"
#ifdef IS_MOBILE_PLATFORM

#include <memory>

#include "machina/core/framework/summary.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/monitoring/metric_def.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"
#else  // IS_MOBILE_PLATFORM

#include <float.h>

#include <map>

#include "machina/core/framework/summary.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/histogram/histogram.h"
#include "machina/core/lib/monitoring/collection_registry.h"
#include "machina/core/lib/monitoring/metric_def.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#endif
// NOLINTBEGIN(misc-unused-using-decls)
namespace machina {
namespace monitoring {

using tsl::monitoring::Buckets;
using tsl::monitoring::Sampler;
using tsl::monitoring::SamplerCell;
}  // namespace monitoring
}  // namespace machina
// NOLINTEND(misc-unused-using-decls)
#endif  // MACHINA_CORE_LIB_MONITORING_SAMPLER_H_
