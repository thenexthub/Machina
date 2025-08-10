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

#include "machina/core/grappler/utils/grappler_test.h"

#include "machina/cc/ops/standard_ops.h"
#include "machina/core/grappler/utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

// TODO(ezhulenev): add tests for all methods in GrapplerTest
class GrapplerTestTest : public GrapplerTest {};

TEST_F(GrapplerTestTest, CompareIdenticalGraphs) {
  machina::Scope s1 = machina::Scope::NewRootScope();
  auto s1_a = ops::Variable(s1.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto s1_b = ops::Variable(s1.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto s1_add = ops::Add(s1.WithOpName("Add_1"), s1_a, s1_b);

  machina::Scope s2 = machina::Scope::NewRootScope();
  auto s2_a = ops::Variable(s2.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto s2_b = ops::Variable(s2.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto s2_add = ops::Add(s2.WithOpName("Add_1"), s2_a, s2_b);

  GraphDef graph1;
  TF_ASSERT_OK(s1.ToGraphDef(&graph1));

  GraphDef graph2;
  TF_ASSERT_OK(s2.ToGraphDef(&graph2));

  CompareGraphs(graph1, graph2);
}

TEST_F(GrapplerTestTest, CheckNodesConnectivity) {
  machina::Scope s = machina::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add_1 = ops::Add(s.WithOpName("Add_1"), a, b);
  auto add_2 = ops::Add(s.WithOpName("Add_2"), add_1, b);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));

  NodeMap node_map(&graph);

  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "a", "Add_1", 0));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "b", "Add_1", 1));
  EXPECT_FALSE(IsNodesDirectlyConnected(node_map, "a", "Add_2", 0));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "b", "Add_2", 1));
}

TEST_F(GrapplerTestTest, CountOpNodes) {
  machina::Scope s = machina::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);

  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_bc = ops::Add(s.WithOpName("Add_bc"), b, c);

  auto mul_ab = ops::Mul(s.WithOpName("Mull_ab"), a, b);
  auto mul_bc = ops::Mul(s.WithOpName("Mull_bc"), a, b);

  InputList inputs{
      Output(add_ab),
      Output(add_bc),
      Output(mul_ab),
      Output(mul_bc),
  };
  auto add_all = ops::AddN(s.WithOpName("Add_all"), inputs);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));

  EXPECT_EQ(2, CountOpNodes(graph, "Add"));
  EXPECT_EQ(2, CountOpNodes(graph, "Mul"));
  EXPECT_EQ(1, CountOpNodes(graph, "AddN"));
  EXPECT_EQ(0, CountOpNodes(graph, "Transpose"));
}

TEST_F(GrapplerTestTest, EvaluateNodes) {
  EnableAllOptimizers();
  machina::Scope s = machina::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output b = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithOpName("mul"), a, b);
  GrapplerItem item;
  item.fetch = {"mul"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].flat<float>()(0), 3.0f);
  EXPECT_EQ(tensors[0].flat<float>()(1), 8.0f);
}

TEST_F(GrapplerTestTest, EvaluateNodesInvalidFetch) {
  EnableAllOptimizers();
  machina::Scope s = machina::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output b = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithOpName("mul"), a, b);
  GrapplerItem item;
  item.fetch = {"no_such_node"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_DEATH(EvaluateNodes(item.graph, item.fetch),
               "Tensor no_such_node:0, specified in either "
               "feed_devices or fetch_devices was not found in the Graph");
}

}  // namespace
}  // namespace grappler
}  // namespace machina
