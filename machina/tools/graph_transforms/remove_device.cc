/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/common_runtime/constant_folding.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/graph/subgraph.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/public/session.h"
#include "machina/tools/graph_transforms/fold_constants_lib.h"
#include "machina/tools/graph_transforms/transform_utils.h"

namespace machina {
namespace graph_transforms {

// Clears the device field of all ops in the graph.
absl::Status RemoveDevice(const GraphDef& input_graph_def,
                          const TransformFuncContext& context,
                          GraphDef* output_graph_def) {
  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    new_node->set_device("");
  }

  return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("remove_device", RemoveDevice);

}  // namespace graph_transforms
}  // namespace machina
