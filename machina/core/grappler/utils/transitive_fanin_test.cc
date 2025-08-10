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

#include "machina/core/grappler/utils/transitive_fanin.h"

#include <vector>

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/grappler/utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

class TransitiveFaninTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, std::vector<string> inputs)
        : name(std::move(name)), inputs(std::move(inputs)) {}
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      NodeDef node_def;
      node_def.set_name(node.name);
      node_def.set_op(node.op);
      for (const string& input : node.inputs) {
        node_def.add_input(input);
      }
      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(TransitiveFaninTest, NoPruning) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}}      //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromSingleTerminalNode) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_FALSE(node_map.NodeExists("5"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromMultipleTerminalNodes) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"2"}},  //
      {"6", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_TRUE(node_map.NodeExists("5"));
  ASSERT_FALSE(node_map.NodeExists("6"));
}

TEST_F(TransitiveFaninTest, InvalidGraphOrTerminalNodes) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"6"}},  //
      {"7", {"8"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  auto s = SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.message(), "Graph does not contain input 6 of node 5.");
  const std::vector<string> invalid_terminal_nodes = {"0", "1", "5"};
  s = SetTransitiveFaninGraph(graph, &output_graph, invalid_terminal_nodes);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.message(), "Graph does not contain terminal node 0.");
}

}  // namespace
}  // namespace grappler
}  // namespace machina
