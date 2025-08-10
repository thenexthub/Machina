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
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/machina/translate/tools/file_tf_mlir_translate.h"
#include "machina/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "machina/compiler/mlir/tools/tf_mlir_translate_cl.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/client/client_library.h"
#include "machina/xla/client/compile_only_client.h"
#include "machina/xla/stream_executor/host/host_platform_id.h"
#include "machina/xla/stream_executor/platform_manager.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/types.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace tf2xla {
namespace v2 {
namespace testing {

using tsl::Status;

static constexpr char kMlirToGraphCompilationCheckName[] =
    "mlir-to-graph-compilation-check";
// Use CPU arbitrarily in order to check that a graph compiles at all
static constexpr char kArbitraryDeviceName[] = "MACHINA_MACHINA_XLA_CPU_JIT";

static Status CompileGraph(machina::Graph* graph,
                           xla::CompileOnlyClient* client) {
  if (!graph || !client) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Invalid graph or client");
  }

  machina::FunctionDefLibrary flib;
  auto flib_def = std::make_unique<machina::FunctionLibraryDefinition>(
      machina::OpRegistry::Global(), flib);

  machina::XlaCompiler::Options options;
  options.device_type = machina::DeviceType(kArbitraryDeviceName);
  options.client = client;
  options.flib_def = flib_def.get();
  machina::XlaCompiler compiler(options);

  std::unique_ptr<machina::Graph> graph_copy(
      new machina::Graph(machina::OpRegistry::Global()));
  machina::CopyGraph(*graph, graph_copy.get());

  machina::XlaCompiler::CompileOptions compile_options;
  machina::XlaCompiler::CompilationResult result;
  return compiler.CompileGraph(compile_options,
                               kMlirToGraphCompilationCheckName,
                               std::move(graph_copy), {}, &result);
}

static mlir::OwningOpRef<mlir::ModuleOp> GraphdefToMlirTranslateFunction(
    toolchain::StringRef input, mlir::MLIRContext* context) {
  machina::GraphdefToMlirOptions options{
      debug_info_file,        xla_compile_device_type,
      prune_unused_nodes,     convert_legacy_fed_inputs,
      graph_as_function,      upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes,
      enable_soft_placement,  set_original_tf_func_name};

  auto module_or = machina::GraphdefToMlirTranslateFunction(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, options, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static mlir::TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static mlir::LogicalResult MlirToGraphTranslateFunction(
    mlir::ModuleOp module, toolchain::raw_ostream& output) {
  if (!module) return mlir::failure();

  machina::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
  confs.export_original_tf_func_name = export_original_tf_func_name;

  std::unique_ptr<machina::FunctionLibraryDefinition> flib_def;
  auto graph =
      std::make_unique<machina::Graph>(machina::OpRegistry::Global());
  absl::flat_hash_set<machina::Node*> control_ret_nodes;
  auto status = machina::tf2xla::v2::ConvertTfExecutorToGraph(
      module, confs, &graph, flib_def.get(), &control_ret_nodes);
  if (!status.ok()) {
    LOG(ERROR) << "Export to Graph failed: " << status;
    return mlir::failure();
  }

  // Use Host platform, which should always exist, to make sure graphs compile.
  auto platform = stream_executor::PlatformManager::PlatformWithId(
      stream_executor::host::kHostPlatformId);
  if (!platform.ok()) {
    return mlir::failure();
  }
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform.value());

  machina::XlaOpRegistry::RegisterCompilationKernels();

  // Verify that the resulting graph can compile.
  if (client.ok() && !CompileGraph(graph.get(), client.value()).ok()) {
    return mlir::failure();
  }

  auto graphdef = std::make_unique<machina::GraphDef>();
  // Print the graph to the output after going through GraphDef conversion.
  // The DumpGraphToFile would do this anyway so just skip straight to it.
  graph->ToGraphDef(graphdef.get());
  output << tsl::LegacyUnredactedDebugString(*graphdef);

  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration mlir_to_graph_translate(
    /*name=*/"mlir-to-graph", /*description=*/"convert mlir to graph",
    MlirToGraphTranslateFunction, [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

static toolchain::LogicalResult MlirToGraphdefTranslateFunction(
    mlir::ModuleOp module, toolchain::raw_ostream& output) {
  if (!module) return mlir::failure();

  machina::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
  confs.export_original_tf_func_name = export_original_tf_func_name;

  machina::FunctionLibraryDefinition flib_def(
      machina::OpRegistry::Global(), machina::FunctionDefLibrary());
  auto graph =
      std::make_unique<machina::Graph>(machina::OpRegistry::Global());
  absl::flat_hash_set<machina::Node*> control_ret_nodes;

  auto status = machina::tf2xla::v2::ConvertTfExecutorToGraph(
      module, confs, &graph, &flib_def, &control_ret_nodes);
  if (!status.ok()) {
    LOG(ERROR) << "Export to Graph failed: " << status;
    return mlir::failure();
  }

  machina::GraphDef graphdef;
  graph->ToGraphDef(&graphdef);
  output << tsl::LegacyUnredactedDebugString(graphdef);
  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration mlir_to_graphdef_translate(
    "mlir-to-graphdef", "mlir-to-graphdef", MlirToGraphdefTranslateFunction,
    [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace machina
