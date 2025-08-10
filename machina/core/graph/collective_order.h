/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_CORE_GRAPH_COLLECTIVE_ORDER_H_
#define MACHINA_CORE_GRAPH_COLLECTIVE_ORDER_H_

#include "machina/core/graph/graph.h"

namespace machina {

enum class GraphCollectiveOrder { kNone, kEdges, kAttrs };

// Introduces a deterministic execution order between potentially concurrent
// CollectiveOps.  This may be used to execute collectives in the same order
// across all workers in a distributed execution, if all workers are executing
// the same graph.
// If `order_type` is `kEdges`, introduce the ordering in the form of explicit
// control edges between collective graph nodes.  If `order_type` is `kAttrs`,
// add an attribute to the node which may be used by collective executor to
// ensure the required ordering.
absl::Status OrderCollectives(Graph* graph, GraphCollectiveOrder order_type);

}  // namespace machina

#endif  // MACHINA_CORE_GRAPH_COLLECTIVE_ORDER_H_
