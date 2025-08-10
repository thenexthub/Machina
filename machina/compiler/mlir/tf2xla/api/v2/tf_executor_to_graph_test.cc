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

#include "machina/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"

#include <stdlib.h>

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/bytes/read_all.h"  // from @riegeli
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
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
using mlir::ModuleOp;
using mlir::OwningOpRef;

std::string TestDataPath() {
  return machina::GetDataDependencyFilepath(
      "machina/compiler/mlir/tf2xla/api/v2/testdata/");
}

class TfExecutorToGraphTest : public ::testing::Test {
 public:
  TfExecutorToGraphTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  absl::StatusOr<OwningOpRef<mlir::ModuleOp>> CreateMlirModule(
      std::string mlir_module_filename) {
    std::string mlir_module_path = TestDataPath() + mlir_module_filename;
    return mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
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

  DialectRegistry registry_;
  MLIRContext context_;
  OwningOpRef<mlir::ModuleOp> mlir_module_;
};

TEST_F(TfExecutorToGraphTest, ConvertMlirToGraphSucceeds) {
  auto valid_executor_module = CreateMlirModule("valid_executor.mlir");
  GraphExportConfig confs;
  absl::flat_hash_set<Node*> control_ret_nodes;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  auto result_graph = std::make_unique<Graph>(flib_def);

  TF_ASSERT_OK(ConvertTfExecutorToGraph(valid_executor_module.value().get(),
                                        confs, &result_graph, &flib_def,
                                        &control_ret_nodes));

  GraphDef result_graphdef;
  result_graph->ToGraphDef(&result_graphdef);
  GraphDef expected_graphdef = CreateGraphDef("valid_graph.txt");
  EXPECT_EQ(result_graphdef.DebugString(), expected_graphdef.DebugString());
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace machina
