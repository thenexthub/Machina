/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_NODE_ORDER_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_NODE_ORDER_H_

#include <functional>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "machina/core/graph/algorithm.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/gtl/array_slice.h"

namespace machina {

struct GroupByDevice {
  std::string operator()(const Node* node) const {
    return node->requested_device();
  }
};

// Performs a topological ordering of nodes.
// This has the property that any child node of a parent node p is emitted
// before p. A grouping function is used to break ties if multiple child nodes
// (of possibly different parents) are ready to be emitted at some point, which
// is when we prefer to stay in the current group. Remaining ties are broken by
// node name.
// The "emit" function is used for outputing the result, and is called once
// for each node.
// This algorithm is O(n * k * log k), with k the largest node degree.
void TopologicalOrdering(
    const Graph& g, const std::function<void(Node*)>& emit,
    const std::function<std::string(Node*)>& get_grouping_key);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_NODE_ORDER_H_
