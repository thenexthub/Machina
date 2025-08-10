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

#ifndef MACHINA_CORE_GRAPPLER_UTILS_CANONICALIZER_H_
#define MACHINA_CORE_GRAPPLER_UTILS_CANONICALIZER_H_

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace grappler {

// Canonicalizes node by performing the following steps
//  - sorting control inputs,
//  - sorting data inputs if the node represents a commutative op.
void CanonicalizeNode(NodeDef* node);

// Canonicalizes all nodes in graph.
void CanonicalizeGraph(GraphDef* graph);

// Compresses Const and HostConstant nodes in the graph to the smallest
// representation possible, either
//   a) truncated repeated field representation, or
//   b) raw serialized byte format.
// Each node is only modified if it is larger than 64 bytes and compression
// reduces its size by more than 50%.
void CompressConstants(GraphDef* graph);

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_CANONICALIZER_H_
