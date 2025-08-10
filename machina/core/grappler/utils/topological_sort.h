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

#ifndef MACHINA_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_
#define MACHINA_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_

#include "absl/types/span.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace grappler {

// TODO(ezhulenev, b/121379902): We should be consistent with GraphTopologyView
// and use `GraphView::Edge` to pass extra dependencies.
struct TopologicalDependency {
  TopologicalDependency(const NodeDef* from, const NodeDef* to)
      : from(from), to(to) {}
  const NodeDef* from;
  const NodeDef* to;
};

// Computes a topological ordering for the graph nodes and outputs nodes in the
// topological order to the `topo_order` output argument.
//
// It's possible to pass additional edges that do not exists in a graph, but
// must be respected when computing graph topological order. Example: Tensorflow
// runtime allows concurrent execution of dequeue/enqueue ops from the same
// queue resource, but we might want to enforce ordering between them.
absl::Status ComputeTopologicalOrder(
    const GraphDef& graph,
    absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<const NodeDef*>* topo_order);
absl::Status ComputeTopologicalOrder(const GraphDef& graph,
                                     std::vector<const NodeDef*>* topo_order);

// Sorts a graph in topological order.
absl::Status TopologicalSort(GraphDef* graph);

// Sorts a graph in topological order and reverse it.
absl::Status ReversedTopologicalSort(GraphDef* graph);

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_
