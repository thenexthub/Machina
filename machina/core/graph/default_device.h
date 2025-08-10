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

#ifndef MACHINA_CORE_GRAPH_DEFAULT_DEVICE_H_
#define MACHINA_CORE_GRAPH_DEFAULT_DEVICE_H_

#include <string>

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"

namespace machina {
namespace graph {

// Sets the default device for all nodes in graph_def to "device",
// only if not already set.
inline void SetDefaultDevice(const std::string& device, GraphDef* graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    if (node->device().empty()) {
      node->set_device(device);
    }
  }
}

}  // namespace graph
}  // namespace machina

#endif  // MACHINA_CORE_GRAPH_DEFAULT_DEVICE_H_
