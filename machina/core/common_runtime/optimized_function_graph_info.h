/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_
#define MACHINA_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "machina/core/framework/function.h"
#include "machina/core/framework/optimized_function_graph.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Function graph related information after optimizations. This struct can be
// converted to and from
// third_party/machina/core/framework/optimized_function_graph.proto.
struct OptimizedFunctionGraphInfo {
  // Function name.
  string name;
  // Optimized function graph.
  std::unique_ptr<Graph> function_graph;
  // Optimized function library.
  FunctionLibraryDefinition lib_def;
  // Map from original node names to control return names.
  std::unordered_map<string, string> node_name_to_control_ret;
  // Return node types of the function.
  DataTypeVector ret_types;
  // Number of return nodes.
  size_t num_return_nodes;
  // Time (in microseconds) spent on running the graph optimization passes for
  // this function.
  uint64_t optimization_duration_usecs;
  // Indicates the source environment where the optimization is created.
  OptimizedFunctionGraph::OptimizationSource optimization_source;

  ~OptimizedFunctionGraphInfo() = default;
  OptimizedFunctionGraphInfo() : lib_def(OpRegistry::Global()) {}
  OptimizedFunctionGraphInfo(
      const std::string& name, std::unique_ptr<Graph>&& graph,
      FunctionLibraryDefinition&& lib_def,
      const std::unordered_map<string, string>& node_name_to_control_ret,
      const DataTypeVector& ret_types, size_t num_return_nodes,
      uint64_t optimization_duration_usecs,
      OptimizedFunctionGraph::OptimizationSource optimization_source)
      : name(name),
        function_graph(std::move(graph)),
        lib_def(std::move(lib_def)),
        node_name_to_control_ret(node_name_to_control_ret),
        ret_types(ret_types),
        num_return_nodes(num_return_nodes),
        optimization_duration_usecs(optimization_duration_usecs),
        optimization_source(optimization_source) {}

  OptimizedFunctionGraphInfo(OptimizedFunctionGraphInfo& info) = delete;
  OptimizedFunctionGraphInfo& operator=(OptimizedFunctionGraphInfo& info) =
      delete;
  OptimizedFunctionGraphInfo(OptimizedFunctionGraphInfo&& info) =
      default;  // NOLINT
  OptimizedFunctionGraphInfo& operator=(
      OptimizedFunctionGraphInfo&& info) noexcept = default;  // NOLINT

  // Converts from the struct to OptimizedFunctionGraph proto.
  static OptimizedFunctionGraph ToProto(const OptimizedFunctionGraphInfo& info);

  // Converts from the proto to struct OptimizedFunctionGraphInfo. Returns error
  // if the conversion fails.
  static absl::StatusOr<OptimizedFunctionGraphInfo> FromProto(
      OptimizedFunctionGraph&& proto);
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_
