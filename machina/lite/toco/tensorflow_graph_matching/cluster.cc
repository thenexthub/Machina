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
#include "machina/lite/toco/machina_graph_matching/cluster.h"

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/lite/toco/machina_graph_matching/cluster_utils.h"

namespace toco {

void Cluster::SetGraphDefInfo(const machina::GraphDef* graph_def) {
  graph_def_ = graph_def;
  for (const machina::NodeDef& node : graph_def_->node()) {
    if (StrContains(node.name(), name_)) {
      nodes_.push_back(&node);
    }
  }
}

bool Cluster::FindClusterInputsAndOutputs() {
  // For every node N in the graph:
  // If N belongs to this cluster C, then each of N's inputs that are not part
  // of C are then inputs of C.
  // If N does not belong to cluster C, then each of N's inputs that belong to C
  // are then outputs of C.
  for (const machina::NodeDef& node : graph_def_->node()) {
    if (StrContains(node.name(), name_)) {
      for (int i = 0; i < node.input_size(); i++) {
        if (!StrContains(node.input(i), name_)) {
          inputs_.push_back(node.input(i));
        }
      }
    } else {
      for (int i = 0; i < node.input_size(); i++) {
        if (StrContains(node.input(i), name_)) {
          outputs_.push_back(node.input(i));
        }
      }
    }
  }
  return (!inputs_.empty()) && (!outputs_.empty());
}

}  // end namespace toco
