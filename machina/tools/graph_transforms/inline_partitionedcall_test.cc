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

#include <algorithm>
#include <string>

#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"
#include "machina/tools/graph_transforms/transform_utils.h"

namespace machina {
namespace graph_transforms {

constexpr char kGraphDefWithPartitionedCall[] =
    "node {\n"
    "  name: \"y\"\n"
    "  op: \"Placeholder\"\n"
    "}\n"
    "node {\n"
    "  name: \"sub/y\"\n"
    "  op: \"Const\"\n"
    "}\n"
    "node {\n"
    "  name: \"PartitionedCall\"\n"
    "  op: \"PartitionedCall\"\n"
    "  input: \"y\"\n"
    "  input: \"sub/y\"\n"
    "  attr {\n"
    "    key: \"f\"\n"
    "    value {\n"
    "      func {\n"
    "        name: \"__inference_simple_add_14\"\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "node {\n"
    "  name: \"add/y\"\n"
    "  op: \"Const\"\n"
    "}\n"
    "node {\n"
    "  name: \"add\"\n"
    "  op: \"AddV2\"\n"
    "  input: \"PartitionedCall\"\n"
    "  input: \"add/y\"\n"
    "}\n"
    "node {\n"
    "  name: \"Identity\"\n"
    "  op: \"Identity\"\n"
    "  input: \"add\"\n"
    "}\n"
    "library {\n"
    "  function {\n"
    "    signature {\n"
    "      name: \"__inference_simple_add_14\"\n"
    "      input_arg {\n"
    "        name: \"x\"\n"
    "        type: DT_FLOAT\n"
    "      }\n"
    "      input_arg {\n"
    "        name: \"y\"\n"
    "        type: DT_FLOAT\n"
    "      }\n"
    "      output_arg {\n"
    "        name: \"identity\"\n"
    "        type: DT_FLOAT\n"
    "      }\n"
    "    }\n"
    "    node_def {\n"
    "      name: \"mul\"\n"
    "      op: \"Mul\"\n"
    "      input: \"x\"\n"
    "      input: \"y\"\n"
    "    }\n"
    "    node_def {\n"
    "      name: \"add/y\"\n"
    "      op: \"Const\"\n"
    "    }\n"
    "    node_def {\n"
    "      name: \"add\"\n"
    "      op: \"AddV2\"\n"
    "      input: \"mul:z:0\"\n"
    "      input: \"add/y:output:0\"\n"
    "    }\n"
    "    node_def {\n"
    "      name: \"Identity\"\n"
    "      op: \"Identity\"\n"
    "      input: \"add:z:0\"\n"
    "    }\n"
    "    ret {\n"
    "      key: \"identity\"\n"
    "      value: \"Identity:output:0\"\n"
    "    }\n"
    "  }\n"
    "}\n";

// Declare here, so we don't need a public header.
absl::Status InlinePartitionedCall(const GraphDef& input_graph_def,
                                   const TransformFuncContext& context,
                                   GraphDef* output_graph_def);

TEST(InlinePartitionedCallTest, Inlining) {
  GraphDef in_graph;
  EXPECT_TRUE(::machina::protobuf::TextFormat::ParseFromString(
      kGraphDefWithPartitionedCall, &in_graph));

  GraphDef result;
  TransformFuncContext context;
  context.input_names = {"y"};
  context.output_names = {"Identity"};
  TF_ASSERT_OK(InlinePartitionedCall(in_graph, context, &result));

  EXPECT_TRUE(std::none_of(
      result.node().cbegin(), result.node().cend(),
      [](const NodeDef& node) { return node.op() == "PartitionedCall"; }));
  EXPECT_EQ(9, result.node().size());
  TF_EXPECT_OK(IsGraphValid(result));
}

}  // namespace graph_transforms
}  // namespace machina
