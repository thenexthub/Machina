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

#include "machina/core/grappler/verifiers/structure_verifier.h"

#include <string>
#include <vector>

#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/graph/validate.h"
#include "machina/core/grappler/utils/topological_sort.h"
#include "machina/core/grappler/verifiers/graph_verifier.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace grappler {

// TODO(ashwinm): Expand this to add more structural checks.
absl::Status StructureVerifier::Verify(const GraphDef& graph) {
  StatusGroup status_group;

  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph.library());
  status_group.Update(machina::graph::ValidateGraphDefAgainstOpRegistry(
      graph, function_library));
  status_group.Update(machina::graph::VerifyNoDuplicateNodeNames(graph));

  std::vector<const NodeDef*> topo_order;
  status_group.Update(ComputeTopologicalOrder(graph, &topo_order));
  return status_group.as_concatenated_status();
}

}  // end namespace grappler
}  // end namespace machina
