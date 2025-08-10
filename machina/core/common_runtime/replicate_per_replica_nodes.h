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

#ifndef MACHINA_CORE_COMMON_RUNTIME_REPLICATE_PER_REPLICA_NODES_H_
#define MACHINA_CORE_COMMON_RUNTIME_REPLICATE_PER_REPLICA_NODES_H_

#include "absl/container/flat_hash_map.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// `composite_device` maps from a virtual device to a set of devices.
// In a function graph, for each node assigned to a composite device
// (representing N devices), replace it with N replicated nodes (one per
// device).
// REQUIREMENTS:
// 1) Each node has been assigned to a device (including composite device).
// 2) Each cluster of nodes assigned to a composite device should include at
// least one "_Arg" node.
// composite device.
// 3) Clusters assigned to different composite devices should have no data
// dependency.
// TODO(b/145922293): Register it as a POST_REWRITE_FOR_EXEC pass.
absl::Status ReplicatePerReplicaNodesInFunctionGraph(
    const absl::flat_hash_map<string, const std::vector<string>*>&
        composite_devices,
    Graph* graph);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_REPLICATE_PER_REPLICA_NODES_H_
