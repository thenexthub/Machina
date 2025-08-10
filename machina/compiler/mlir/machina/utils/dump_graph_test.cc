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

#include "machina/compiler/mlir/machina/utils/dump_graph.h"

#include "machina/core/graph/graph.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"
#include "machina/core/util/dump_graph.h"

namespace machina {
namespace {

void ExpectHasSubstr(const string& s, const string& expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

void ExpectHasNoSubstr(const string& s, const string& expected) {
  EXPECT_FALSE(absl::StrContains(s, expected))
      << "'" << s << "' should not contain '" << expected << "'";
}

// WritableFile that simply concats into string.
class StringWritableFile : public WritableFile {
 public:
  explicit StringWritableFile(string* str) : str_(*str) {}

  absl::Status Append(absl::string_view data) override {
    absl::StrAppend(&str_, data);
    return absl::OkStatus();
  }

  absl::Status Close() override { return absl::OkStatus(); }

  absl::Status Flush() override { return absl::OkStatus(); }

  absl::Status Name(absl::string_view* result) const override {
    *result = "(string)";
    return absl::OkStatus();
  }

  absl::Status Sync() override { return absl::OkStatus(); }

  absl::Status Tell(int64_t* position) override {
    return errors::Unimplemented("Stream not seekable");
  }

 private:
  string& str_;
};

TEST(Dump, TextualIrToFileSuccess) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  UseMlirForGraphDump(MlirDumpConfig());
  string ret = DumpGraphToFile("tir", graph);
  ASSERT_EQ(ret, io::JoinPath(testing::TmpDir(), "tir.mlir"));

  string actual;
  TF_ASSERT_OK(ReadFileToString(Env::Default(), ret, &actual));
}

TEST(Dump, TextualIrWithOptions) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_ASSERT_OK(NodeBuilder("A", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Finalize(&graph, &node));

  string actual;
  StringWritableFile file(&actual);
  TF_ASSERT_OK(DumpTextualIRToFile(MlirDumpConfig().emit_location_information(),
                                   graph, /*flib_def=*/nullptr, &file));

  string expected_substr = R"(loc(#loc))";
  ExpectHasSubstr(actual, expected_substr);
}

TEST(Dump, DumpToTFG) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  string actual;
  StringWritableFile file(&actual);

  TF_ASSERT_OK(DumpTextualIRToFile(
      MlirDumpConfig().emit_dialect(MlirDumpConfig::Dialect::kTFG), graph,
      /*flib_def=*/nullptr, &file));

  string expected_substr("tfg.graph");
  ExpectHasSubstr(actual, expected_substr);

  string not_expected_substr("tf_executor.island");
  ExpectHasNoSubstr(actual, not_expected_substr);
}

}  // namespace
}  // namespace machina
