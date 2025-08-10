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

#include "machina/core/util/dump_graph.h"

#include "absl/strings/match.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/op.h"
#include "machina/core/graph/graph.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/status.h"

namespace machina {
namespace {

TEST(DumpGraph, DumpGraphToFileSuccess) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpGraphToFile("graph", graph);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "graph.pbtxt"));
  ret = DumpGraphToFile("graph", graph);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "graph_1.pbtxt"));

  GraphDef gdef;
  TF_ASSERT_OK(ReadTextProto(
      Env::Default(), io::JoinPath(testing::TmpDir(), "graph.pbtxt"), &gdef));
  string read, written;
  gdef.AppendToString(&read);
  graph.ToGraphDefDebug().AppendToString(&written);
  EXPECT_EQ(read, written);
}

TEST(DumpGraph, DumpGraphToFileNoEnvPrefix) {
  Graph graph(OpRegistry::Global());
  unsetenv("TF_DUMP_GRAPH_PREFIX");
  string ret = DumpGraphToFile("graph", graph);
  EXPECT_TRUE(absl::StrContains(ret, "TF_DUMP_GRAPH_PREFIX not specified"));
}

TEST(DumpGraph, DumpFunctionDefToFileSuccess) {
  FunctionDef fdef;
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpFunctionDefToFile("function", fdef);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "function.pbtxt"));
}

TEST(DumpGraph, DumpProtoToFileSuccess) {
  NodeDef ndef_in;
  ndef_in.set_name("foo");
  ndef_in.set_op("bar");
  ndef_in.add_input("baz");
  ndef_in.set_device("cpu:0");

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  setenv("TF_DUMP_GRAPH_FMT", "TXT", 1);
  string expected_filepath = io::JoinPath(testing::TmpDir(), "node_def.pbtxt");
  string actual_filepath = DumpProtoToFile("node_def", ndef_in);
  EXPECT_EQ(expected_filepath, actual_filepath);

  NodeDef ndef_out;
  TF_ASSERT_OK(ReadTextProto(Env::Default(), expected_filepath, &ndef_out));
  EXPECT_EQ(ndef_in.DebugString(), ndef_out.DebugString());

  setenv("TF_DUMP_GRAPH_FMT", "BIN", 1);
  string ret = DumpProtoToFile("node_def", ndef_in);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "node_def_1.pb"));
  TF_ASSERT_OK(ReadBinaryProto(Env::Default(), ret, &ndef_out));
  EXPECT_EQ(ndef_out.DebugString(), ndef_in.DebugString());

  setenv("TF_DUMP_GRAPH_FMT", "unknown", 1);
  ret = DumpProtoToFile("node_def", ndef_in);
  EXPECT_TRUE(absl::StrContains(ret, "Unknown format"));
}

}  // namespace
}  // namespace machina
