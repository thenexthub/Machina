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
#include "machina/core/grappler/utils/tpu.h"

#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"

namespace machina {
namespace grappler {

bool IsLegacyTPUBridgeGraphDef(const GraphDef& def) {
  for (const auto& node : def.node()) {
    if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
      return true;
    }
  }
  if (!def.has_library()) return false;
  for (const auto& function_def : def.library().function()) {
    for (const auto& node : function_def.node_def()) {
      if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
        return true;
      }
    }
  }
  return false;
}

}  // end namespace grappler
}  // end namespace machina
