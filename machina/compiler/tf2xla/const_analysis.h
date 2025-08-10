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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLACONST_ANALYSIS_H_
#define MACHINA_COMPILER_TF2MACHINA_XLACONST_ANALYSIS_H_

#include <vector>

#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Backwards dataflow analysis that finds nodes in a graph that must be
// compile-time constants for us to be able to lower the graph to XLA.
//
// The indices of the arguments to `graph` that must be constant are returned in
// `compile_time_const_arg_indices`, if `compile_time_const_arg_indices` is not
// null.
//
// The ids of the nodes in `graph` that must be constant are returned in
// `compile_time_const_nodes`, if `compile_time_const_nodes` is not null.
//
// If `edge_filter` is non-null, only propagate const-ness along edges for which
// `edge_filter` returns true.
absl::Status BackwardsConstAnalysis(
    const Graph& g, std::vector<bool>* compile_time_const_arg_indices,
    std::vector<bool>* compile_time_const_nodes,
    FunctionLibraryRuntime* flib_runtime,
    std::function<bool(const Edge&)> edge_filter_input = nullptr);

// Given an op kernel and function library runtime, return all the indices of
// inputs that need to be compile time constant.
absl::Status GetCompileTimeConstInputs(const OpKernel* op_kernel,
                                       std::vector<int>* const_input_idxs,
                                       FunctionLibraryRuntime* flib_runtime);
}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLACONST_ANALYSIS_H_
