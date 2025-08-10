/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_

#include <unordered_map>

#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/clusters/cluster.h"
#include "machina/core/grappler/costs/cost_estimator.h"
#include "machina/core/grappler/grappler_item.h"

namespace machina {
namespace grappler {

// Compute the earliest time at which the execution of each node in the graph
// can complete.
// In our estimation, we ensure that each node takes at least one nanosecond to
// execute: therefore the execution times can be used to derive a topological
// ordering of the graph (at least as long as there is no loop in the graph).
absl::Status EstimateEarliestExecutionTimes(
    const GrapplerItem& item, const Cluster* cluster,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* execution_times);

// Compute the time by which the execution of each node must complete to ensure
// the subsequent nodes can still be executed by the times predicted by the
// EstimateEarliestExecutionTimes function.
absl::Status EstimateRequiredTimes(
    const GrapplerItem& item, const Cluster* cluster,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* required_times);

}  // namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_
