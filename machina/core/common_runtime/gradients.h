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

#ifndef MACHINA_CORE_COMMON_RUNTIME_GRADIENTS_H_
#define MACHINA_CORE_COMMON_RUNTIME_GRADIENTS_H_

#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/gtl/array_slice.h"

namespace machina {

// Represents the output of 'node' at 'index'.
struct NodeOut {
  Node* node;
  int index;

  // Returns the string name that represents the output of this node.
  string name() const;
  // Returns the data type of the output of this node.
  DataType dtype() const;
};

// NOTE: This API is a work in progress and will likely be changing frequently.
//
// Given initial gradient-node outputs 'y_grad_node_outputs' (which compute the
// symbolic partial derivatives of some loss function 'L' w.r.t the node outputs
// 'y_node_outputs'), adds gradient nodes to 'graph' that compute the symbolic
// partial derivatives of 'L' w.r.t the node outputs 'x_node_outputs'.
//
// REQUIRES: Each node in 'x_node_outputs' to be unique, and so to have a single
// output (this restriction will be removed in a subsequent change).

// TODO(andydavis) Add symbolic gradient support for general graphs (the current
// implementation only supports gradients for functions). In particular,
// the nodes in 'x_nodes' are currently restricted to have one output.

absl::Status AddSymbolicGradients(absl::Span<const NodeOut> y_node_outputs,
                                  absl::Span<const NodeOut> x_node_outputs,
                                  absl::Span<const NodeOut> y_grad_node_outputs,
                                  std::vector<NodeOut>* x_grad_node_outputs,
                                  Graph* graph);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_GRADIENTS_H_
