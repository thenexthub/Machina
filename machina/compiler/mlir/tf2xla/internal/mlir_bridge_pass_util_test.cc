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

#include "machina/compiler/mlir/tf2xla/internal/mlir_bridge_pass_util.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/function_ops.h"
#include "machina/cc/ops/tpu_functional_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/tf2xla/tf2xla_defs.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/function_testlib.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/platform/enable_tf2_utils.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

namespace {

// Produce a valid graph with a resource-type input.
FunctionDef PassThroughResource() {
  return FunctionDefHelper::Define(
      /*function_name=*/"PassThroughResource",
      /*arg_def=*/{"in: resource"},
      /*ret_def=*/{"out: resource"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"out"}, "Identity", {"in"}, {{"T", DataType::DT_RESOURCE}}}});
}

TEST(IsSupportedByNonReplicatedBridge, NonReplicatedGraph) {
  const FunctionDef& fd = PassThroughResource();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  machina::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_RESOURCE, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_RESOURCE})
          .Attr("Tout", {DT_RESOURCE})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kMustCompileAttr), true);

  TF_ASSERT_OK(root.ToGraph(&graph));

  // Required for passing the PS server parameter check.
  for (Node* node : graph.nodes()) {
    node->set_assigned_device_name("/job:ps/replica:0/task:0/device:GPU:0");
  }

  EXPECT_TRUE(
      IsSupportedByNonReplicatedBridge(graph, /*function_library=*/nullptr));
}

TEST(IsSupportedByReplicatedBridge, ReplicatedGraph) {
  const FunctionDef& fd = test::function::XTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  machina::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kTpuReplicateAttr), "cluster");

  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(
      IsSupportedByReplicatedBridge(graph, /*function_library=*/nullptr));
}

TEST(IsSupportedByReplicatedBridge, ReplicatedModule) {
  const char* const code = R"mlir(
func.func @entry_func_1(%arg0: tensor<i32>) -> tensor<i32> attributes {tf.entry_function = {}} {
  %0 = "tf.Identity"(%arg0) {_tpu_replicate = "cluster"} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}
)mlir";
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(IsSupportedByReplicatedBridge(*module));
}

TEST(HasTPUPartitionedCallOpInModule, HasTPUPartitionedCallModule) {
  const char* const code = R"mlir(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() {
    %outputs_0 = "tf.TPUOrdinalSelector"() {device = ""} : () -> tensor<?xi32>
    "tf.TPUPartitionedCall"(%outputs_0) {f = @reachable_func} : (tensor<?xi32>) -> ()
    func.return
  }
  func.func @reachable_func() {
    func.return
  }
}
)mlir";
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(HasTPUPartitionedCallOpInModule(*module));
}

TEST(HasTPUPartitionedCallOpInModule, HasNotTPUPartitionedCallModule) {
  const char* const code = R"mlir(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() {
    "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @reachable_func} : () -> ()
    func.return
  }
  func.func @reachable_func() {
    func.return
  }
}
)mlir";
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  EXPECT_FALSE(HasTPUPartitionedCallOpInModule(*module));
}

TEST(IsInferenceGraph, GraphContrainsTPUPartitionedCall) {
  FunctionDef fd = FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloat",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"},
           "Const",
           {},
           {{"value", test::AsScalar<int32>(2)}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });

  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(IsInferenceGraph(graph, /*function_library=*/nullptr));
}

TEST(IsInferenceGraph, GraphDoesNotContrainTPUPartitionedCall) {
  FunctionDef fd = FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloat",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"},
           "Const",
           {},
           {{"value", test::AsScalar<int32>(2)}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });

  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(IsInferenceGraph(graph, /*function_library=*/nullptr));
}

TEST(IsInferenceGraph, FlibDefIsNotNullptrAndContainsTPUPartitionedCall) {
  FunctionDef fd = FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloat",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"},
           "Const",
           {},
           {{"value", test::AsScalar<int32>(2)}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
          {{"tpu_op"}, "TPUPartitionedCall", {}, {{"Tout", DT_FLOAT}}},
      });

  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(IsInferenceGraph(graph, /*function_library=*/&flib_def));
}

}  // namespace

}  // namespace machina
