/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"

#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/control_flow_ops.h"
#include "machina/cc/ops/functional_ops.h"
#include "machina/cc/ops/tpu_functional_ops.h"
#include "machina/cc/ops/tpu_replication_ops.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/enable_tf2_utils.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

namespace {

REGISTER_OP("OneRefOutput").Output("y: Ref(float)");

FunctionDef XTimesTwo() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimesTwoFloat() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
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
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });
}

FunctionDef XTimesTwoFloatRef() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloatRef",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64_REF}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64_REF}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });
}

Node* FromNodeDef(absl::string_view name, absl::string_view node_type,
                  int num_inputs, DataType dt, Graph& graph) {
  auto builder = NodeDefBuilder(name, node_type);
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(absl::StrCat("node_", i), i, dt);
  }

  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));

  absl::Status s;
  Node* node = graph.AddNode(node_def, &s);
  TF_CHECK_OK(s);
  return node;
}

TEST(SupportedGraphTest, SupportedGraphReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  machina::set_tf2_execution(true);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InvalidGraphTest, InvalidFuncBodyReturnsTrue) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwo();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwo");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  // The call to XTimesTwo is invalid (missing an attribute), so we expect the
  // graph to be unsupported.
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesReturnsTrue) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  // Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  machina::set_tf2_execution(true);
  const std::vector<int32> shape_array{2, 2};
  auto shape = TensorShape();
  TF_ASSERT_OK(TensorShapeUtils::MakeShape(shape_array, &shape));
  Output value = Output(
      FromNodeDef("value", "OneRefOutput", 0, DT_FLOAT_REF, *root.graph()));

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, NoRefVariablesCalleeFuncReturnsFalse) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesInCalleeFunctionReturnsTrue) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloatRef();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloatRef");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesInExternalCalleeFunctionReturnsTrue) {
  machina::set_tf2_execution(true);
  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloatRef();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloatRef");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/&flib_def, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeEagerRuntimeReturnsTrue) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeTFRTBridgeReturnsTrue) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeDirectSessionReturnsFalse) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(ControlFlowTest, ContainsV1ControlFlowReturnsTrue) {
  machina::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(ControlFlowTest, TFRTContainsV1ControlFlowReturnsTrue) {
  machina::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF1ReturnsTrue) {
  machina::set_tf2_execution(false);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseV1CompatBridgeReturnTrue) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(false);

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueV1CompatBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseTfrtNominalBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(false);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueTfrtNominalBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = machina::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = machina::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = machina::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)machina::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseNominalBridgeReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(false);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueNominalBridgeReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = machina::ops::Placeholder(root.WithOpName("input"), DT_UINT8);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  machina::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(UnsupportedOpTest,
     InfeedDequeueTupleWithTPUReplicatedCoreAttrNotSupported) {
  machina::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);

  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  // Build InfeedDequeueTuple node with TPU_REPLICATED_CORE Attr
  auto builder = NodeDefBuilder("InfeedDequeueTuple", "InfeedDequeueTuple");
  builder.Attr("dtypes", DT_FLOAT);
  builder.Attr("shapes", 1);
  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));
  absl::Status s;
  Node* node_InfeedDequeueTuple = (*root.graph()).AddNode(node_def, &s);
  node_InfeedDequeueTuple->set_requested_device(
      "/device:TPU_REPLICATED_CORE:0");
  TF_CHECK_OK(s);
  ASSERT_NE(node_InfeedDequeueTuple, nullptr);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/true));
}

TEST(ManualControlDependencyTest,
     TPUReplicatedCoreWithManualControlDependencyReturnsFalse) {
  machina::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);

  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  auto metadata = machina::ops::TPUReplicateMetadata(root, 2);
  metadata.operation.node()->AddAttr("_has_manual_control_dependencies", true);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/true));
}

TEST(InferenceTest,
     ContainsInferenceNodeTPUReplicatedCoreDirectSessionReturnsFalse) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest,
     ContainsInferenceNodeTPUReplicatedCoreEagerRuntimeReturnsTrue) {
  machina::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, TF2ExecutionFalseV1CompatBridgeReturnFalse) {
  machina::set_tf2_execution(false);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, V1CompatBridgeVariableRefReturnTrue) {
  machina::set_tf2_execution(false);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      machina::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  machina::set_tf2_execution(true);
  const std::vector<int32> shape_array{2, 2};
  auto shape = TensorShape();
  TF_ASSERT_OK(TensorShapeUtils::MakeShape(shape_array, &shape));
  Output value = Output(
      FromNodeDef("value", "OneRefOutput", 0, DT_FLOAT_REF, *root.graph()));

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/machina::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

}  // namespace

}  // namespace machina
