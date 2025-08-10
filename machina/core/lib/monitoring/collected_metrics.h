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

// Standard format in which the metrics are collected, before being exported.
// These are to be used only by the CollectionRegistry and exporters which
// collect metrics using the CollectionRegistry.

#ifndef MACHINA_CORE_LIB_MONITORING_COLLECTED_METRICS_H_
#define MACHINA_CORE_LIB_MONITORING_COLLECTED_METRICS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "machina/xla/tsl/lib/monitoring/collected_metrics.h"
#include "machina/core/framework/summary.pb.h"
#include "machina/core/lib/monitoring/metric_def.h"
#include "machina/core/lib/monitoring/types.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace machina {
namespace monitoring {
using tsl::monitoring::CollectedMetrics;
using tsl::monitoring::MetricDescriptor;
using tsl::monitoring::Point;
using tsl::monitoring::PointSet;
}  // namespace monitoring
}  // namespace machina
// NOLINTEND(misc-unused-using-decls)
#endif  // MACHINA_CORE_LIB_MONITORING_COLLECTED_METRICS_H_
