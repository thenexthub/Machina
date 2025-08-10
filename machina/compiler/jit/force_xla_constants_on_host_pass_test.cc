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

#include "machina/compiler/jit/force_xla_constants_on_host_pass.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/ops/functional_ops.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/compiler/jit/compilability_check_util.h"
#include "machina/compiler/jit/defs.h"
#include "machina/compiler/jit/test_util.h"
#include "machina/core/common_runtime/function.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/function_testlib.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/graph/graph_def_builder.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"
#include "machina/core/public/version.h"

namespace machina {
namespace {

absl::Status ForceXlaConstantsOnHost(const Scope& s,
                                     FunctionLibraryDefinition* flib_def,
                                     std::unique_ptr<Graph>* result) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.env = Env::Default();
  options.graph = &graph;
  options.session_options = &session_options;
  options.flib_def = flib_def;
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  ForceXlaConstantsOnHostPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return absl::OkStatus();
}

TEST(ForceXlaConstantsOnHostPassTest, Simple) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;

  FunctionDef called_func =
      FunctionDefHelper::Create("TransposeCall",
                                /*in_def=*/{"a:float", "b:int32"},
                                /*out_def=*/{"c:float"}, {},
                                {{{"t0"},
                                  "Transpose",
                                  {"a", "b"},
                                  {
                                      {"T", DT_FLOAT},
                                      {"Tperm", DT_INT32},
                                  }}},
                                {{"c", "t0:y:0"}});

  AttrValue true_attribute;
  true_attribute.set_b(true);
  (*called_func.mutable_attr())[kXlaMustCompileAttr] = true_attribute;
  *library.add_function() = called_func;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), library);
  Output in = ops::Placeholder(root, DT_FLOAT);
  Output perm = ops::Const(root, {3, 1, 2, 0});

  NameAttrList b_name_attr;
  b_name_attr.set_name("TransposeCall");
  ops::PartitionedCall call(root.WithOpName("call"), {in, perm}, {DT_FLOAT},
                            b_name_attr);
  call.output.front().node()->AddAttr(kXlaMustCompileAttr, true);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(ForceXlaConstantsOnHost(root, &flib_def, &graph));

  bool found = false;
  for (Node* node : graph->nodes()) {
    if (CanCreateXlaKernel(node->def())) {
      EXPECT_FALSE(found);
      found = true;
      std::vector<int32> hostmem_attr;
      EXPECT_TRUE(TryGetNodeAttr(node->def(), "_input_hostmem", &hostmem_attr));
      EXPECT_EQ(hostmem_attr.size(), 1);
      EXPECT_EQ(hostmem_attr[0], 1);
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace
}  // namespace machina
