/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_CORE_LIB_MONITORING_CELL_READER_H_
#define MACHINA_CORE_LIB_MONITORING_CELL_READER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "machina/xla/tsl/lib/monitoring/cell_reader.h"
#include "machina/core/lib/monitoring/cell_reader-inl.h"
#include "machina/core/lib/monitoring/collected_metrics.h"
#include "machina/core/lib/monitoring/metric_def.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace machina {
namespace monitoring {
namespace testing {
using tsl::monitoring::testing::CellReader;
}  // namespace testing
}  // namespace monitoring
}  // namespace machina
// NOLINTEND(misc-unused-using-decls)
#endif  // MACHINA_CORE_LIB_MONITORING_CELL_READER_H_
