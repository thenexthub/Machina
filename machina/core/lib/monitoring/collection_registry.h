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

#ifndef MACHINA_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
#define MACHINA_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_

#include "machina/xla/tsl/lib/monitoring/collection_registry.h"
// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "machina/core/platform/platform.h"
// clang-format on
// We use a null implementation for mobile platforms.
#ifdef IS_MOBILE_PLATFORM

#include <functional>
#include <map>
#include <memory>

#include "machina/core/lib/monitoring/metric_def.h"
#include "machina/core/platform/macros.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace machina {
namespace monitoring {
using tsl::monitoring::CollectionRegistry;
using tsl::monitoring::MetricCollector;
using tsl::monitoring::MetricCollectorGetter;
}  // namespace monitoring
}  // namespace machina
// NOLINTEND(misc-unused-using-decls)
#else  // !defined(IS_MOBILE_PLATFORM)

#include <functional>
#include <map>
#include <memory>
#include <utility>

#include "machina/core/framework/summary.pb.h"
#include "machina/core/lib/monitoring/collected_metrics.h"
#include "machina/core/lib/monitoring/metric_def.h"
#include "machina/core/lib/monitoring/types.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace monitoring {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::monitoring::CollectionRegistry;
using tsl::monitoring::Exporter;
using tsl::monitoring::MetricCollector;
using tsl::monitoring::MetricCollectorGetter;
using tsl::monitoring::exporter_registration::ExporterRegistration;
using tsl::monitoring::internal::Collector;
namespace test_util {
class CollectionRegistryTestAccess;
}  // namespace test_util
// NOLINTEND(misc-unused-using-decls)
}  // namespace monitoring
}  // namespace machina

#endif  // IS_MOBILE_PLATFORM

#endif  // MACHINA_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
