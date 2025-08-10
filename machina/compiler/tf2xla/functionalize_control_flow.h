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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_FUNCTIONALIZE_CONTROL_FLOW_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_FUNCTIONALIZE_CONTROL_FLOW_H_

#include "machina/compiler/tf2xla/functionalize_control_flow_util.h"
#include "machina/xla/status_macros.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"

namespace machina {

const char kFunctionalizeControlFlowFailureMessage[] =
    "Failed to functionalize Control Flow V1 ops. Consider using Control "
    "Flow V2 ops instead. See "
    "https://www.machina.org/api_docs/python/tf/"
    "compat/v1/enable_control_flow_v2.";

// Transformation that converts tf.while_loop() loops into functional While
// operators and tf.cond() conditionals into function If operators, suitable for
// XLA compilation.
//
// If `node_filter` is defined, then only loops and conditions for whose
// nodes `node_filter` returns true are functionalized.

// If `include_functions` is true, then loops and conditions inside of functions
// that are associated with nodes in `graph` (e.g., a function called from a
// node in `graph`) are also functionalized, otherwise they are not.
// This also handles transitive cases, e.g., a function body will be
// functionalized when it is called in another function that is called by some
// node in `graph` (and so on). The node filter also applies here.
//
// Precondition:
// For any node in a loop or condition for which `node_filter` returns true,
// all nodes inside of the same loop or condition must also return true
// (including nodes in other nested loops and conditions inside of that loop or
// condition).
// This means that a "not to be functionalized" loop or condition is not allowed
// inside a "to be functionalized" loop or condition.
//
// The user of this function is responsible for using a node filter that
// satisfies the above conditions.
absl::Status FunctionalizeControlFlow(Graph* graph,
                                      FunctionLibraryDefinition* library,
                                      const NodeFilter& node_filter = {},
                                      bool include_functions = false);

absl::Status FunctionalizeControlFlowForGraphDef(
    GraphDef* graph_def, FunctionLibraryDefinition* library,
    const NodeFilter& node_filter = {}, bool include_functions = false);

// Rewrites the graph by turning V1 control flow structure
// (Switch/Merge/etc.) into V2 control flow structure (If/While), only modifies
// functions that will be executed by XLA.
class FunctionalizeControlFlowForXlaPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_FUNCTIONALIZE_CONTROL_FLOW_H_
