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

#include "machina/compiler/mlir/tf2xla/api/v1/compile_tf_graph.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/transforms/set_tpu_infeed_layout.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/compiler/mlir/machina/utils/translate_utils.h"
#include "machina/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "machina/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "machina/compiler/tf2xla/layout_util.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/xla/client/compile_only_client.h"
#include "machina/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "machina/xla/mlir_hlo/mhlo/IR/register.h"
#include "machina/xla/service/hlo.pb.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/status_macros.h"
#include "machina/xla/tsl/framework/device_type.h"
#include "machina/xla/tsl/lib/monitoring/sampler.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/versions.pb.h"
#include "machina/core/lib/monitoring/counter.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/profile_utils/cpu_utils.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"
#include "machina/core/tpu/tpu_compile.h"
#include "machina/core/util/debug_data_dumper.h"

namespace machina {
namespace tf2xla {
namespace v1 {

using ::machina::tpu::FunctionToHloArgs;
using ::machina::tpu::GuaranteedConsts;
using ::machina::tpu::MlirToHloArgs;
using ::machina::tpu::ShardingAndIndex;

auto* phase2_bridge_compilation_status =
    machina::monitoring::Counter<1>::New(
        "/machina/core/tf2xla/api/v1/"
        "phase2_compilation_status", /*metric_name*/
        "Tracks the compilation status of the non-mlir bridge",
        /* metric description */ "status" /* metric label */);

auto* phase2_bridge_compilation_time = tsl::monitoring::Sampler<1>::New(
    {"/machina/core/tf2xla/api/v1/phase2_compilation_time",
     "The wall-clock time spent on executing graphs in milliseconds.",
     "configuration"},
    // Power of 1.5 with bucket count 45 (> 23 hours)
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 45)});

// There were no MLIR ops so the old bridge was called successfully.
constexpr char kOldBridgeNoMlirSuccess[] = "kOldBridgeNoMlirSuccess";
// There were no MLIR ops so the old bridge was called but it failed.
constexpr char kOldBridgeNoMlirFailure[] = "kOldBridgeNoMlirFailure";

namespace {

// Time the execution of kernels (in CPU cycles). Meant to be used as RAII.
struct CompilationTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }

  int64_t ElapsedCyclesInMilliseconds() {
    std::chrono::duration<double> duration =
        profile_utils::CpuUtils::ConvertClockCycleToTime(ElapsedCycles());

    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};

// Populates input_output_alias field in the HLO Module proto.
absl::Status PopulateInputOutputAliasing(
    mlir::func::FuncOp main_fn,
    XlaCompiler::CompilationResult* compilation_result, bool use_tuple_args) {
  constexpr char kAliasingAttr[] = "tf.aliasing_output";
  toolchain::SmallDenseMap<unsigned, unsigned> output_to_input_alias;
  unsigned num_arguments = main_fn.getNumArguments();
  for (unsigned arg_index = 0; arg_index < num_arguments; ++arg_index) {
    if (auto aliasing_output = main_fn.getArgAttrOfType<mlir::IntegerAttr>(
            arg_index, kAliasingAttr))
      output_to_input_alias[aliasing_output.getInt()] = arg_index;
  }

  if (output_to_input_alias.empty()) return absl::OkStatus();

  xla::HloModuleProto* module_proto =
      compilation_result->computation->mutable_proto();
  absl::StatusOr<xla::ProgramShape> program_shape_or_status =
      compilation_result->computation->GetProgramShape();
  TF_RET_CHECK(program_shape_or_status.ok());

  xla::ProgramShape& program_shape = program_shape_or_status.value();
  if (!program_shape.result().IsTuple())
    return errors::Internal("Expect result to have tuple shape");

  xla::HloInputOutputAliasConfig config(program_shape.result());
  for (auto alias : output_to_input_alias) {
    if (use_tuple_args) {
      TF_RETURN_IF_ERROR(config.SetUpAlias(
          xla::ShapeIndex({alias.first}), 0, xla::ShapeIndex({alias.second}),
          xla::HloInputOutputAliasConfig::AliasKind::kMayAlias));
    } else {
      TF_RETURN_IF_ERROR(config.SetUpAlias(
          xla::ShapeIndex({alias.first}), alias.second, xla::ShapeIndex({}),
          xla::HloInputOutputAliasConfig::AliasKind::kMayAlias));
    }
  }
  *module_proto->mutable_input_output_alias() = config.ToProto();
  return absl::OkStatus();
}

bool failed(const absl::Status& status) { return !status.ok(); }

// Transforms the given module to be suitable for export to TensorFlow GraphDef
// and then exports all functions to the given library.
absl::Status PrepareAndExportToLibrary(mlir::ModuleOp module,
                                       FunctionLibraryDefinition* flib_def) {
  // Pass pipeline is defined here instead of leveraging the phase one export
  // pipeline because only the functional to executor dialect conversion and
  // breakup islands passes are common between the export pipeline and here.
  // Reconsider this if there is more commonality in the future with more
  // passes.
  mlir::PassManager manager(module.getContext());
  applyTensorflowAndCLOptions(manager);
  manager.addPass(mlir::TF::CreatePrepareTpuComputationForTfExportPass());
  manager.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
  manager.addPass(mlir::TF::CreateTFShapeInferencePass());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  manager.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  if (VLOG_IS_ON(2)) {
    toolchain::StringRef module_name = toolchain::StringRef();
    constexpr const char* kDebugGroupBridgePhase2 =
        "v1_prepare_and_export_to_library";
    internal::EnablePassIRPrinting(manager, kDebugGroupBridgePhase2,
                                   module_name);
  }

  auto prepare_status = manager.run(module);
  auto diag_handler_status = diag_handler.ConsumeStatus();
  // There are cases where the scoped diagnostic handler catches a failure that
  // the running of the passes does not. That causes the handler to throw if
  // it is not consumed.
  if (failed(prepare_status) || failed(diag_handler_status)) {
    return diag_handler_status;
  }

  GraphExportConfig config;
  config.export_entry_func_to_flib = true;
  absl::flat_hash_set<Node*> control_ret_nodes;
  return machina::tf2xla::v2::ConvertTfExecutorToGraph(
      module, config, /*graph=*/nullptr, flib_def, &control_ret_nodes);
}

absl::Status CompileTFFunctionWithoutMlir(
    FunctionToHloArgs function_computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_funcs,
    const std::vector<machina::TensorShape>& arg_shapes,
    const DeviceType& device_type,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result) {
  absl::Status comp_status = CompileTFFunctionToHlo(
      *function_computation.flib_def, function_computation.graph_def_version,
      shape_determination_funcs, arg_shapes, device_type,
      function_computation.guaranteed_constants, *function_computation.function,
      metadata, client, arg_core_mapping, per_core_arg_shapes, use_tuple_args,
      compilation_result);
  if (comp_status.ok()) {
    phase2_bridge_compilation_status->GetCell(kOldBridgeNoMlirSuccess)
        ->IncrementBy(1);
  } else {
    phase2_bridge_compilation_status->GetCell(kOldBridgeNoMlirFailure)
        ->IncrementBy(1);
  }

  return comp_status;
}

absl::Status CompileMLIRTFFunction(
    mlir::ModuleOp mlir_module, const tpu::TPUCompileMetadataProto& metadata,
    bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_funcs,
    const std::vector<machina::TensorShape>& arg_shapes,
    const DeviceType& device_type,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result) {
  if (!mlir::SetTPUInfeedLayout(mlir_module))
    return errors::Internal("Failed to set layouts attribute");

  if (VLOG_IS_ON(2)) {
    machina::DumpMlirOpToFile("legalize_with_old_bridge", mlir_module);
  }
  constexpr char kEntryFuncName[] = "main";
  auto main_fn = mlir_module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!main_fn) {
    return errors::Internal(
        "TPU compile op requires module with a entry function main");
  }

  // Export functions to the library.
  auto flib_def = std::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), FunctionDefLibrary());
  TF_RETURN_IF_ERROR(PrepareAndExportToLibrary(mlir_module, flib_def.get()));

  if (VLOG_IS_ON(2)) {
    machina::DumpMlirOpToFile("legalize_with_old_bridge_post_transform",
                                 mlir_module);
  }
  VersionDef versions;
  if (mlir::failed(ExtractTfVersions(mlir_module, &versions))) {
    return errors::Internal(
        "module attribute in _TPUCompileMlir op is missing tf versions.");
  }

  NameAttrList func;
  func.set_name(kEntryFuncName);
  GuaranteedConsts consts;

  *compilation_result = {};

  TF_RETURN_IF_ERROR(CompileTFFunctionToHlo(
      *flib_def, versions.producer(), shape_determination_funcs, arg_shapes,
      device_type, consts, func, metadata, client, arg_core_mapping,
      per_core_arg_shapes, use_tuple_args, compilation_result));

  return PopulateInputOutputAliasing(main_fn, compilation_result,
                                     use_tuple_args);
}

absl::Status CompileMLIRTFFunction(
    tpu::MlirToHloArgs mlir_computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_funcs,
    const std::vector<machina::TensorShape>& arg_shapes,
    const DeviceType& device_type,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result) {
  if (mlir_computation.mlir_module_op.has_value()) {
    return CompileMLIRTFFunction(
        mlir_computation.mlir_module_op.value(), metadata, use_tuple_args,
        shape_determination_funcs, arg_shapes, device_type, arg_core_mapping,
        per_core_arg_shapes, client, compilation_result);
  }

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  TF_RETURN_IF_ERROR(DeserializeMlirModule(mlir_computation.mlir_module,
                                           &context, &mlir_module));

  return CompileMLIRTFFunction(*mlir_module, metadata, use_tuple_args,
                               shape_determination_funcs, arg_shapes,
                               device_type, arg_core_mapping,
                               per_core_arg_shapes, client, compilation_result);
}

}  // namespace

absl::Status CompileTensorflowGraphToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_funcs,
    const std::vector<machina::TensorShape>& arg_shapes,
    tsl::DeviceType device_type,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result) {
  LOG_FIRST_N(INFO, 1) << "Compiling MLIR computation to XLA HLO using the "
                          "old (non-MLIR) tf2xla bridge";

  CompilationTimer timer;
  *compilation_result = {};
  bool has_mlir = computation.index() == 0;

  std::string mlir_string = has_mlir ? "has_mlir" : "has_function_to_hlo";
  const std::string kBridgePhase2Config =
      absl::StrCat("graph_old_bridge_", mlir_string);

  if (has_mlir) {
    TF_RETURN_IF_ERROR(CompileMLIRTFFunction(
        std::get<0>(computation), metadata, use_tuple_args,
        shape_determination_funcs, arg_shapes, device_type, arg_core_mapping,
        per_core_arg_shapes, client, compilation_result));

  } else {
    FunctionToHloArgs function_computation = std::get<1>(computation);
    TF_RETURN_IF_ERROR(CompileTFFunctionWithoutMlir(
        function_computation, metadata, use_tuple_args,
        shape_determination_funcs, arg_shapes, device_type, arg_core_mapping,
        per_core_arg_shapes, client, compilation_result));
  }

  phase2_bridge_compilation_time->GetCell(kBridgePhase2Config)
      ->Add(timer.ElapsedCyclesInMilliseconds());

  return absl::OkStatus();
}

};  // namespace v1
};  // namespace tf2xla
};  // namespace machina
