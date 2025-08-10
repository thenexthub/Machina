/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/compiler/jit/force_xla_constants_on_host_pass.h"

#include "machina/compiler/jit/compilability_check_util.h"
#include "machina/compiler/jit/defs.h"
#include "machina/core/common_runtime/optimization_registry.h"

namespace machina {

absl::Status ForceXlaConstantsOnHostPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  OptimizerOptions opts;
  auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, options.flib_def, opts);
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  for (Node* node : graph->nodes()) {
    if (CanCreateXlaKernel(node->def())) {
      const FunctionBody* fbody = nullptr;
      std::vector<int> constant_arg_indices;
      std::vector<int> resource_arg_indices;

      NameAttrList function;
      TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node->def(), &function));

      // Force all constants to be on the host memory.
      TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
          flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));
      VLOG(3) << "Found constant arg indices: "
              << absl::StrJoin(constant_arg_indices, ", ");

      node->AddAttr("_input_hostmem", constant_arg_indices);
    }
  }
  return absl::OkStatus();
}

}  // namespace machina
