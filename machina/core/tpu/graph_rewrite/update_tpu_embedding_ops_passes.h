/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// Rewrites ConfigureTPUEmbedding Op into nodes which set up TPUEmbedding.

#ifndef MACHINA_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_
#define MACHINA_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_

#include <map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/status.h"

namespace machina {

class UpdateTPUEmbeddingEnqueueOrdinalPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

class UpdateTPUEmbeddingModePass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;

  static absl::Status GetEnqueueOpsFromGraph(
      Graph* graph, absl::flat_hash_map<Node*, bool>* enqueue);
  static absl::Status UpdateGraphEnqueueOp(bool training, Graph* graph,
                                           Node* enqueue);
  static absl::Status GetEnqueueOpsFromFunctionDef(
      FunctionDef* function, std::map<int, bool>* enqueue);
  static absl::Status UpdateFunctionDefEnqueueOp(int enqueue, bool training,
                                                 FunctionDef* function,
                                                 bool* updated);
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_
