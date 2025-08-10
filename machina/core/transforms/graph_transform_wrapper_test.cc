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
#include "machina/core/transforms/graph_transform_wrapper.h"

#include <memory>

#include "absl/strings/match.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/common_runtime/graph_def_builder_util.h"
#include "machina/core/graph/graph.h"
#include "machina/core/graph/graph_def_builder.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/test.h"

namespace mlir {
namespace {

// Testing pass that deletes a single op from the Graph. This assumes the
// graph created below.
struct TestPass : public PassWrapper<TestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPass)

  TestPass() = default;
  StringRef getArgument() const final { return "test"; }
  void runOnOperation() override {
    Operation* del;
    getOperation()->walk([&](Operation* op) {
      if (op->getName().getStringRef() != "tfg.TestInput") return;
      del = *op->getResult(0).getUsers().begin();
    });
    del->erase();
  }
};

}  // namespace
}  // namespace mlir

REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestRelu").Input("i: float").Output("o: float");
REGISTER_OP("NoOp");

TEST(GraphTransformWrapper, ReplacedGraph) {
  machina::Graph graph(machina::OpRegistry::Global());
  {
    machina::GraphDefBuilder b(
        machina::GraphDefBuilder::kFailImmediately);
    machina::Node* input =
        machina::ops::SourceOp("TestInput", b.opts().WithName("in"));
    machina::ops::UnaryOp("TestRelu", machina::ops::NodeOut(input, 0),
                             b.opts().WithName("n1"));
    machina::ops::UnaryOp("TestRelu", machina::ops::NodeOut(input, 1),
                             b.opts().WithName("n2"));
    TF_EXPECT_OK(machina::GraphDefBuilderToGraph(b, &graph));
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::tfg::TFGraphDialect>();

  auto create_pass = [&]() { return std::make_unique<mlir::TestPass>(); };

  TF_QCHECK_OK(mlir::tfg::RunTransformOnGraph(&graph, {create_pass}));

  EXPECT_EQ(4, graph.num_nodes());
  EXPECT_TRUE(
      absl::StrContains(graph.ToGraphDefDebug().ShortDebugString(), "\"n2\""));
}
