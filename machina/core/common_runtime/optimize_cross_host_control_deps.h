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

#ifndef MACHINA_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_
#define MACHINA_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_

#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Optimize the graph by reducing cross-host control output edges.
// Once we find any nodes in the graph having not less than
// `cross_host_edges_threshold` control output edges in one host, we create
// a `NoOp` node in the destination host to proxy the control edges between the
// oringal node and the destination control output nodes.
absl::Status OptimizeCrossHostControlOutputEdges(
    Graph* graph, int cross_host_edges_threshold);

// Optimize the graph by reducing cross-host data output edges.
// Once we find any nodes in the graph having not less than
// `cross_host_edges_threshold` data output edges in one host, we create
// a `IdentityN` node in the destination host to proxy the data edges between
// the original node and the destination output nodes.
absl::Status OptimizeCrossHostDataOutputEdges(Graph* graph,
                                              int cross_host_edges_threshold);

// Optimize the graph by reducing cross-host control input edges.
// Once we find any nodes in the graph having not less than
// `cross_host_edges_threshold` control input edges in one host, we create
// a `NoOp` node in the source host to proxy the control edges between the
// source control input nodes and oringal node.
absl::Status OptimizeCrossHostControlInputEdges(Graph* graph,
                                                int cross_host_edges_threshold);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_
