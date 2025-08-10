/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/core/tfrt/utils/debug/node_io_dump_rewriter.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/common_runtime/function_body.h"
#include "machina/core/common_runtime/function_def_utils.h"
#include "machina/core/common_runtime/function_utils.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/graph_to_functiondef.h"
#include "machina/core/framework/op.h"
#include "machina/core/graph/graph.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace machina {
namespace tfrt_stub {
namespace {

absl::StatusOr<std::string> GetDumpDir(absl::string_view dump_dir) {
  if (!dump_dir.empty()) return std::string(dump_dir);
  const char* prefix = getenv("TF_DUMP_GRAPH_PREFIX");
  if (prefix != nullptr) return std::string(prefix);
  return errors::InvalidArgument("TF_DUMP_GRAPH_PREFIX not specified");
}

absl::Status InsertDumpOpsForNode(Graph& graph, Node& node,
                                  absl::string_view dump_dir) {
  auto insert = [&](bool is_input, const std::vector<const Edge*> edges) {
    for (const Edge* edge : edges) {
      if (edge->IsControlEdge()) continue;
      // For each edge, insert a dump node.
      Node* dump_node;
      TF_RETURN_IF_ERROR(
          NodeBuilder(absl::StrCat(edge->src()->name(), "/", edge->src_output(),
                                   "/debug_identity"),
                      "DebugIdentityV3")
              .Attr("io_of_node", node.name())
              .Attr("is_input", is_input)
              .Attr("io_index",
                    is_input ? edge->dst_input() : edge->src_output())
              .Attr("tensor_name",
                    absl::StrCat(edge->src()->name(), ":", edge->src_output()))
              .Attr("debug_urls", {absl::StrCat("file://", dump_dir)})
              .Input(edge->src(), edge->src_output())
              .Finalize(&graph, &dump_node));
      TF_RETURN_IF_ERROR(
          graph.UpdateEdge(dump_node, 0, edge->dst(), edge->dst_input()));
    }
    return absl::OkStatus();
  };

  // Make a copy of the edges to avoid modifying edges while iterating.
  TF_RETURN_IF_ERROR(insert(/*is_input=*/true,
                            {node.in_edges().begin(), node.in_edges().end()}));
  TF_RETURN_IF_ERROR(insert(
      /*is_input=*/false, {node.out_edges().begin(), node.out_edges().end()}));
  return absl::OkStatus();
}

}  // namespace

absl::Status InsertDumpOps(
    Graph& graph, const absl::flat_hash_set<std::string>& nodes_to_dump,
    absl::string_view dump_dir) {
  TF_ASSIGN_OR_RETURN(auto dir, GetDumpDir(dump_dir));
  auto insert = [&](Graph& graph) {
    for (Node* node : graph.op_nodes()) {
      if (nodes_to_dump.contains(node->name())) {
        TF_RETURN_IF_ERROR(InsertDumpOpsForNode(graph, *node, dir));
      }
    }
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(insert(graph));
  for (const auto& fname : graph.flib_def().ListFunctionNames()) {
    // Convert fdef to graph.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *graph.flib_def().Find(fname), AttrSlice(), &graph.flib_def(), &fbody));
    // Insert dump nodes.
    TF_RETURN_IF_ERROR(insert(*fbody->graph));
    // Convert graph to fdef.
    FunctionDef new_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*fbody->graph, fname, &new_fdef));
    TF_RETURN_IF_ERROR(
        graph.mutable_flib_def()->ReplaceFunction(fname, new_fdef));
  }
  return absl::OkStatus();
}

absl::Status InsertDumpOps(
    MetaGraphDef& meta_graph_def,
    const absl::flat_hash_set<std::string>& nodes_to_dump,
    absl::string_view dump_dir) {
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph({}, meta_graph_def.graph_def(), &graph));
  TF_RETURN_IF_ERROR(InsertDumpOps(graph, nodes_to_dump, dump_dir));
  graph.ToGraphDef(meta_graph_def.mutable_graph_def());
  return absl::OkStatus();
}

}  // namespace tfrt_stub
}  // namespace machina
