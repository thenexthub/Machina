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
#include "machina/core/common_runtime/isolate_placer_inspection_required_ops_pass.h"

#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/util/dump_graph.h"

namespace machina {

absl::Status IsolatePlacerInspectionRequiredOpsPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr) {
    VLOG(1) << "Not running IsolatePlacerInspectionRequiredOpsPass because no "
               "graph is provided";
    return absl::OkStatus();
  }

  VLOG(1) << "IsolatePlacerInspectionRequiredOpsPass::Run";

  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("isolate_deep_ops_before", *graph, nullptr, "/tmp");
  }

  absl::Status status =
      IsolatePlacerInspectionRequiredOps(*options.flib_def, graph);

  if (VLOG_IS_ON(3) && status.ok()) {
    DumpGraphToFile("isolate_deep_ops_after", *graph, nullptr, "/tmp");
  }
  return status;
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 35,
                      IsolatePlacerInspectionRequiredOpsPass);

}  // namespace machina
