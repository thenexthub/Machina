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

#ifndef MACHINA_CORE_GRAPPLER_UTILS_SCC_H_
#define MACHINA_CORE_GRAPPLER_UTILS_SCC_H_

#include <unordered_map>
#include "machina/core/framework/graph.pb.h"
#include "machina/core/grappler/inputs/utils.h"
#include "machina/core/lib/io/path.h"

namespace machina {
namespace grappler {

// Computes modified strongly connected components:
// All nodes that are not part of a loop are assigned the special -1 id
// All nodes that are part of at least one loop are assigned a positive
// component id: if 2 nodes v and w are reachable from one another (i.e. if they
// belong to the same scc), they'll be assigned the same id, otherwise they'll
// be assigned distinct ids. *num_components is set to the number of distinct
// ids.
void StronglyConnectedComponents(
    const GraphDef& graph, std::unordered_map<const NodeDef*, int>* components,
    int* num_components);

// Returns the number of individual loops present in the graph, and populate the
// 'loops' argument with the collection of loops (denoted by their loop ids) a
// node is part of. Loops ids are arbitrary.
int IdentifyLoops(const GraphDef& graph,
                  std::unordered_map<const NodeDef*, std::vector<int>>* loops);

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_SCC_H_
