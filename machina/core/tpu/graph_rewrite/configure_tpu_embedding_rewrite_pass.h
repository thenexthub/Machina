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

#ifndef MACHINA_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_
#define MACHINA_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_

#include "absl/status/status.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/status.h"

namespace machina {

// TODO(shizhiw): Clean up embedding related code from
//  distributed_tpu_configuration_rewrite_pass.cc.
// Replaces dummy ConfigureTPUEmbedding Ops assigned to TPU_SYSTEM
// devices with nodes which will set up TPU Embedding.
class ConfigureTPUEmbeddingRewritePass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_
