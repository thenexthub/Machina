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

#include "machina/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor.h"

#include <stdlib.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/bytes/read_all.h"  // from @riegeli
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/resource_loader.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace tf2xla {
namespace v2 {
namespace {

using mlir::DialectRegistry;
using mlir::MLIRContext;

constexpr char kGraphWithFlibDefFileName[] = "graph_with_flib_def.txt";

std::string TestDataPath() {
  return machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tf2xla/api/v2/testdata/");
}

class GraphToTfExecutorTest : public ::testing::Test {
 public:
  GraphToTfExecutorTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  GraphDef CreateGraphDef(std::string graphdef_filename) {
    std::string file_path = TestDataPath() + graphdef_filename;
    std::string contents;
    GraphDef graph_def;
    auto status = riegeli::ReadAll(riegeli::FdReader(file_path), contents);
    if (!status.ok()) {
      return graph_def;
    }
    tsl::protobuf::TextFormat::ParseFromString(contents, &graph_def);
    return graph_def;
  }

  int CountNumberOfFunctionsInModule(mlir::ModuleOp module) {
    int count = 0;
    for (auto unused : module.getOps<mlir::func::FuncOp>()) {
      (void)unused;  // Avoid unused variable warning
      count++;
    }
    return count;
  }

  DialectRegistry registry_;
  MLIRContext context_;
};

TEST_F(GraphToTfExecutorTest, BasicConvertGraphToTfExecutorPasses) {
  Graph graph(OpRegistry::Global());
  GraphDebugInfo debug_info;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphImportConfig specs;
  GraphDef graph_def = CreateGraphDef("valid_graph.txt");
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  TF_ASSERT_OK(
      ConvertGraphToTfExecutor(graph, debug_info, flib_def, specs, &context_));
}

TEST_F(
    GraphToTfExecutorTest,
    ConvertGraphToTfExecutorConvertAllFunctionsTrueConvertsAllFunctionsInFlibDef) {
  Graph graph(OpRegistry::Global());
  GraphDebugInfo debug_info;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDef graph_def = CreateGraphDef(kGraphWithFlibDefFileName);
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  GraphImportConfig specs;
  specs.convert_all_functions_to_mlir = true;

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> result =
      ConvertGraphToTfExecutor(graph, debug_info, graph.flib_def(), specs,
                               &context_);

  // should equal main + 4 functions in flib_def
  ASSERT_EQ(CountNumberOfFunctionsInModule(result->get()), 5);
}

TEST_F(
    GraphToTfExecutorTest,
    ConvertGraphToTfExecutorConvertAllFunctionsFalseOnlyConvertsFunctionsReferencedInGraph) {
  Graph graph(OpRegistry::Global());
  GraphDebugInfo debug_info;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDef graph_def = CreateGraphDef(kGraphWithFlibDefFileName);
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  GraphImportConfig specs;
  specs.convert_all_functions_to_mlir = false;

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> result =
      ConvertGraphToTfExecutor(graph, debug_info, graph.flib_def(), specs,
                               &context_);

  // should equal main + 2 functions referenced by nodes in the graph via the
  // "f" attr.
  ASSERT_EQ(CountNumberOfFunctionsInModule(result->get()), 3);
}

TEST_F(GraphToTfExecutorTest,
       ConvertGraphToTfExecutorPopulatesTfNameToMlirNameMap) {
  Graph graph(OpRegistry::Global());
  GraphDebugInfo debug_info;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDef graph_def = CreateGraphDef(kGraphWithFlibDefFileName);
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  GraphImportConfig specs;
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;

  TF_ASSERT_OK(ConvertGraphToTfExecutor(graph, debug_info, graph.flib_def(),
                                        specs, &context_,
                                        &tf_name_to_mlir_name));

  std::unordered_set<std::string> result_set;
  for (const auto& pair : tf_name_to_mlir_name) {
    result_set.insert(pair.first);
  }
  // Converted functions referenced by nodes in the graph via the
  // "f" attr. These are before they are converted to their corresponding MLIR
  // name.
  std::unordered_set<std::string> expected_set = {
      "__inference__traced_save_45", "__inference__traced_restore_57"};
  EXPECT_EQ(result_set, expected_set);
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace machina
