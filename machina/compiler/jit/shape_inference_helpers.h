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

#ifndef MACHINA_COMPILER_JIT_SHAPE_INFERENCE_HELPERS_H_
#define MACHINA_COMPILER_JIT_SHAPE_INFERENCE_HELPERS_H_

#include <vector>

#include "machina/core/graph/graph.h"

namespace machina {

// Helper class to temporarily remove, then replace, the back edges in a
// graph. Simple algorithms for shape inference don't work with cycles, and this
// class can be used to remove cycles before running inference and replace them
// after. Correct usage requires exactly one call to Remove(), followed by any
// number of calls to RemovedEdges() and at most one call to Replace(). The call
// to Replace() is optional if the graph will be discarded without being
// executed, e.g., if it is being used purely for a shape inference pass.
class BackEdgeHelper {
 public:
  struct BackEdge {
    const Edge* edge;
    Node* src;
    int src_output;
    Node* dst;
    int dst_input;
  };

  BackEdgeHelper() = default;
  // Disallows copy and assign.
  BackEdgeHelper(const BackEdgeHelper& other) = delete;
  BackEdgeHelper& operator=(const BackEdgeHelper& other) = delete;

  // Temporarily removes all the back edges in graph.
  absl::Status Remove(Graph* graph);

  // Gets the list of removed edges.
  const std::vector<BackEdge>& RemovedEdges() const;

  // Replaces the back edges removed by a prior call to Remove.
  absl::Status Replace();

 private:
  Graph* graph_ = nullptr;  // not owned
  std::vector<BackEdge> back_edges_;
  // Set once Replace has been called.
  bool replaced_ = false;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_SHAPE_INFERENCE_HELPERS_H_
