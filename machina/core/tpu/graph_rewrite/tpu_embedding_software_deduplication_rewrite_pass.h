/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_
#define MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_

#include "absl/status/status.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/status.h"

namespace machina {

// Rewrites the graph and function defs in the specified
// GraphOptimizationPassOptions object for software deduplication.
//
// For the graph, groups the RecvTPUEmbeddingActivations and
// SendTPUEmbeddingGradients nodes by their _tpu_replicate attribute. For each
// such group:
// 1. Inserts a XlaRecvTPUEmbeddingDeduplicationData node into the graph.
// 2. Replaces the public RecvTPUEmbeddingActivations node (if present) with the
//    internal XlaRecvTPUEmbeddingActivations node.
// 3. Replaces the public SendTPUEmbeddingGradients node (if present) with the
//    internal XlaSendTPUEmbeddingGradients node.
// 4. Connects the outputs of the XlaRecvTPUEmbeddingDeduplicationData node with
//    the inputs of the XlaRecvTPUEmbeddingActivations and
//    XlaSendTPUEmbeddingGradients nodes.
//
// Iterates through the list of functions in the specified
// GraphOptimizationPassOptions object. Performs the same steps 1-4 specified
// above for each function.
//
// If multiple RecvTPUEmbeddingActivations nodes or SendTPUEmbeddingGradients
// nodes are present in the same function or in the same _tpu_replicate group,
// an InvalidArgument error is returned to the caller.
class TPUEmbeddingSoftwareDeduplicationRewritePass :
    public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_
