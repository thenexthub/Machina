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

#include "machina/core/framework/function.h"
#include "machina/core/framework/graph_def_util.h"
#include "machina/tools/graph_transforms/transform_utils.h"

namespace machina {
namespace graph_transforms {

// Sets any parameters not specified in a node to their defaults.
absl::Status AddDefaultAttributes(const GraphDef& input_graph_def,
                                  const TransformFuncContext& context,
                                  GraphDef* output_graph_def) {
  // Find all of the ops that are currently defined.
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(),
                                    input_graph_def.library()));
  // Works in-place, so copy over the original graph.
  *output_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(output_graph_def, *flib_def, 0));
  return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("add_default_attributes", AddDefaultAttributes);

}  // namespace graph_transforms
}  // namespace machina
