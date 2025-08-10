/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/graph_analyzer/graph_analyzer.h"
#include "machina/core/grappler/utils/transitive_fanin.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/protobuf/meta_graph.pb.h"

namespace machina {
namespace grappler {
namespace graph_analyzer {

// Dies on failure.
static void LoadModel(const string& filename,
                      machina::MetaGraphDef* metagraph) {
  LOG(INFO) << "Loading model from " << filename;
  absl::Status st;
  st = ReadBinaryProto(Env::Default(), filename, metagraph);
  if (!st.ok()) {
    LOG(WARNING) << "Failed to read a binary metagraph: " << st;
    st = ReadTextProto(Env::Default(), filename, metagraph);
    if (!st.ok()) {
      LOG(FATAL) << "Failed to read a text metagraph: " << st;
    }
  }
}

// Prune the graph to only keep the transitive fanin part with respect to a set
// of train ops (if provided).
void MaybePruneGraph(const machina::MetaGraphDef& metagraph,
                     machina::GraphDef* graph) {
  std::vector<string> fetch_nodes;
  for (const auto& fetch :
       metagraph.collection_def().at("train_op").node_list().value()) {
    LOG(INFO) << "Fetch node: " << fetch;
    fetch_nodes.push_back(fetch);
  }
  if (fetch_nodes.empty()) {
    *graph = metagraph.graph_def();
  } else {
    std::vector<const machina::NodeDef*> fanin_nodes;
    TF_CHECK_OK(machina::grappler::ComputeTransitiveFanin(
        metagraph.graph_def(), fetch_nodes, &fanin_nodes));
    for (const machina::NodeDef* node : fanin_nodes) {
      *(graph->add_node()) = *node;
    }
    LOG(INFO) << "Pruned "
              << metagraph.graph_def().node_size() - graph->node_size()
              << " nodes. Original graph size: "
              << metagraph.graph_def().node_size()
              << ". New graph size: " << graph->node_size() << ".";
  }
}

void GraphAnalyzerTool(const string& file_name, int n) {
  if (n < 1) {
    LOG(FATAL) << "Invalid subgraph size " << n << ", must be at least 1";
  }

  machina::MetaGraphDef metagraph;
  LoadModel(file_name, &metagraph);
  machina::GraphDef graph;
  MaybePruneGraph(metagraph, &graph);
  machina::grappler::graph_analyzer::GraphAnalyzer analyzer(graph, n);
  LOG(INFO) << "Running the analysis";
  absl::Status st = analyzer.Run();
  if (!st.ok()) {
    LOG(FATAL) << "Analysis failed: " << st;
  }

  LOG(INFO) << "Printing the result";
  st = analyzer.OutputSubgraphs();
  if (!st.ok()) {
    LOG(FATAL) << "Failed to print the result: " << st;
  }

  LOG(INFO) << "Completed";
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace machina
