/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_
#define MACHINA_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_

#include <unordered_map>
#include <vector>

#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace grappler {

// Computes the transitive fanin of the graph based on reachability from the
// specified terminal nodes. Returns the set of nodes comprising the
// transitive fanin into fanin_nodes. Optionally returns a map of name->node
// for that graph into name_to_fanin_node if that is not set to nullptr.
absl::Status ComputeTransitiveFanin(
    const GraphDef& graph, const std::vector<string>& terminal_nodes,
    std::unordered_map<string, const NodeDef*>* name_to_fanin_node,
    std::vector<const NodeDef*>* fanin_nodes);

absl::Status ComputeTransitiveFanin(const GraphDef& graph,
                                    const std::vector<string>& terminal_nodes,
                                    std::vector<const NodeDef*>* fanin_nodes);

// Creates output_graph from input_graph using the transitive fanin from the
// specified terminal nodes. Returns error if the input_graph is deemed
// structurally invalid.
absl::Status SetTransitiveFaninGraph(const GraphDef& input_graph,
                                     GraphDef* output_graph,
                                     const std::vector<string>& terminal_nodes);

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_
