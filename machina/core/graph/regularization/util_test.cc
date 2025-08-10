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

#include "machina/core/graph/regularization/util.h"

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/platform/test.h"

namespace machina::graph_regularization {

namespace {

GraphDef CreateTestGraph() {
  GraphDef graph_def;
  NodeDef* node = graph_def.add_node();
  node->set_name("name1");
  node->set_op("op1");
  node = graph_def.add_node();
  node->set_name("name2");
  node->set_op("op2");
  return graph_def;
}

TEST(UtilTest, TestGetSuffixUID) { EXPECT_EQ(*GetSuffixUID("foo_32"), 32); }

TEST(UtilTest, TestGetSuffixUID64Bit) {
  EXPECT_EQ(*GetSuffixUID("foo_2209431640"), 2209431640);
}

TEST(UtilTest, TestGetSuffixUIDInvalid) {
  EXPECT_FALSE(GetSuffixUID("foo").ok());
}

TEST(FingerprintingTest, TestComputeHash) {
  GraphDef graph_def = CreateTestGraph();
  EXPECT_EQ(ComputeHash(graph_def), 4870331646167591885);
}
}  // namespace
}  // namespace machina::graph_regularization
