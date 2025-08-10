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

#ifndef MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_
#define MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_

#include "absl/status/status.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/status.h"

namespace machina {

// Adds a new TensorFlow graph node, with the output convention matching most TF
// code rather than the order used by Graph::AddNode().
absl::Status AddNode(const NodeDef& n_def, Node** n, Graph* graph);

// Replaces one TensorFlow graph node with another (specified by a NodeDef),
// moving all the edges.
absl::Status ReplaceNode(const NodeDef& to_def, Node* from, Node** to,
                         Graph* graph);

}  // namespace machina

#endif  // MACHINA_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_
