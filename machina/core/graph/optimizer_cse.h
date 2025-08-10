/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

// An optimization pass that performs common subexpression elimination.

#ifndef MACHINA_CORE_GRAPH_OPTIMIZER_CSE_H_
#define MACHINA_CORE_GRAPH_OPTIMIZER_CSE_H_

#include <sys/types.h>
#include "machina/core/graph/graph.h"

namespace machina {

// Perform common-subexpression elimination on the graph "*g".  If
// "consider_fn" is not nullptr, then only nodes for which
// consider_fn(node) returns true will be considered for combining
// during the common subexpression elimination.
//
// Returns true if and only if 'g' is mutated.
extern bool OptimizeCSE(Graph* g,
                        const std::function<bool(const Node*)>& consider_fn);

}  // namespace machina

#endif  // MACHINA_CORE_GRAPH_OPTIMIZER_CSE_H_
