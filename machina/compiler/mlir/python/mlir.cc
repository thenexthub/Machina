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

#include "machina/compiler/mlir/python/mlir.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/c/eager/c_api.h"
#include "machina/c/eager/tfe_context_internal.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_status_helper.h"
#include "machina/cc/saved_model/bundle_v2.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/transforms/tf_saved_model_passes.h"
#include "machina/compiler/mlir/machina/translate/import_model.h"
#include "machina/compiler/mlir/machina/translate/mlir_import_options.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/machina/translate/tf_mlir_translate.h"
#include "machina/compiler/mlir/machina/translate/tools/parsers.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/compiler/mlir/machina/utils/import_utils.h"
#include "machina/compiler/mlir/machina/utils/mlprogram_util.h"
#include "machina/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor.h"
#include "machina/compiler/mlir/tf2xla/transforms/passes.h"
#include "machina/xla/mlir/framework/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/common_runtime/function_body.h"
#include "machina/core/common_runtime/function_def_utils.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/types.h"

namespace machina {

namespace {
// All the passes we will make available to Python by default.
// TODO(tf): this should be sharded instead of being monolithic like that.
static void RegisterPasses() {
  static bool unique_registration = [] {
    mlir::registerAllPasses();
    mlir::registerTensorFlowPasses();
    mlir::TFDevice::registerTensorFlowDevicePasses();
    mlir::mhlo::registerAllMhloPasses();
    // These are in compiler/mlir/xla and not part of the above MHLO
    // passes.
    mlir::mhlo::registerTfXlaPasses();
    mlir::mhlo::registerLegalizeTFPass();
    mlir::quant::stablehlo::registerBridgePasses();
    mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
    mlir::xla_framework::registerXlaFrameworkPasses();
    machina::RegisterMlProgramPasses();
    return true;
  }();
  (void)unique_registration;
}

// Runs pass pipeline `pass_pipeline` on `module` if `pass_pipeline` is not
// empty.
std::string RunPassPipelineOnModule(mlir::ModuleOp module,
                                    const std::string& pass_pipeline,
                                    bool show_debug_info, TF_Status* status) {
  RegisterPasses();
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(module.getContext());
    std::string error;
    toolchain::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Invalid pass_pipeline: " + error_stream.str()).c_str());
      return "// error";
    }

    mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext());
    if (failed(pm.run(module))) {
      tsl::Set_TF_Status_from_Status(status, statusHandler.ConsumeStatus());
      return "// error";
    }
  }
  return MlirModuleToString(module, show_debug_info);
}

}  // anonymous namespace

static std::string ImportGraphDefImpl(const std::string& proto,
                                      const std::string& pass_pipeline,
                                      bool show_debug_info,
                                      GraphDebugInfo& debug_info,
                                      GraphImportConfig& specs,
                                      TF_Status* status) {
  GraphDef graphdef;
  auto s = machina::LoadProtoFromBuffer(proto, &graphdef);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  GraphConstructorOptions options;
  Graph graph(OpRegistry::Global());
  absl::Status graph_status = ConvertGraphDefToGraph(options, graphdef, &graph);
  auto module = machina::tf2xla::v2::ConvertGraphToTfExecutor(
      graph, debug_info, graph.flib_def(), specs, &context);
  if (!module.ok() || !graph_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportFunction(const std::string& functiondef_proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, TFE_Context* tfe_context,
                           TF_Status* status) {
  FunctionDef functiondef;
  auto s = machina::LoadProtoFromBuffer(functiondef_proto, &functiondef);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  const std::string& function_name = functiondef.signature().name();
  EagerContext* cpp_context = ContextFromInterface(unwrap(tfe_context));
  FunctionLibraryDefinition& flib_def = *cpp_context->FuncLibDef();
  const machina::FunctionDef* fdef = flib_def.Find(function_name);
  if (fdef == nullptr) {
    s = machina::errors::NotFound("Cannot find function ", function_name);
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  std::unique_ptr<machina::FunctionBody> fbody;
  s = FunctionDefToBodyHelper(*fdef, machina::AttrSlice(), &flib_def,
                              &fbody);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);

  machina::GraphImportConfig specs;
  specs.graph_func_name = fbody->record->fdef().signature().name();
  specs.enable_shape_inference = false;
  specs.graph_as_function = true;
  for (const auto* control_ret_node : fbody->control_ret_nodes)
    specs.control_outputs.push_back(control_ret_node->name());
  auto module = machina::tf2xla::v2::ConvertGraphToTfExecutor(
      *fbody->graph, {}, flib_def, specs, &context);
  if (!module.ok()) {
    tsl::Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportGraphDef(const std::string& proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, TF_Status* status) {
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ImportGraphDef(const std::string& proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, absl::string_view input_names,
                           absl::string_view input_data_types,
                           absl::string_view input_data_shapes,
                           absl::string_view output_names, TF_Status* status) {
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  auto s = ParseInputArrayInfo(input_names, input_data_types, input_data_shapes,
                               &specs.inputs);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  if (!output_names.empty()) {
    specs.outputs = absl::StrSplit(output_names, ',');
  }
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ExperimentalConvertSavedModelToMlir(
    const std::string& saved_model_path, const std::string& exported_names_str,
    bool show_debug_info, TF_Status* status) {
  // Load the saved model into a SavedModelV2Bundle.

  machina::SavedModelV2Bundle bundle;
  auto load_status =
      machina::SavedModelV2Bundle::Load(saved_model_path, &bundle);
  if (!load_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelV2Bundle to an MLIR module.

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  auto module_or = ConvertSavedModelToMlir(
      &bundle, &context, absl::Span<std::string>(exported_names));
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*std::move(module_or).value(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlirLite(
    const std::string& saved_model_path, const std::string& exported_names_str,
    const std::string& tags, bool upgrade_legacy, bool show_debug_info,
    TF_Status* status) {
  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);

  machina::MLIRImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  auto module_or = SavedModelSignatureDefsToMlirImportLite(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names),
      &context, import_options);
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.value(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlir(
    const std::string& saved_model_path, const std::string& exported_names_str,
    const std::string& tags, bool lift_variables,
    bool include_variables_in_initializers, bool upgrade_legacy,
    bool show_debug_info, TF_Status* status) {
  // Load the saved model into a SavedModelBundle.

  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  machina::SavedModelBundle bundle;
  auto load_status =
      machina::LoadSavedModel({}, {}, saved_model_path, tag_set, &bundle);
  if (!load_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelBundle to an MLIR module.
  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  machina::MLIRImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  import_options.lift_variables = lift_variables;
  import_options.include_variables_in_initializers =
      include_variables_in_initializers;
  auto module_or =
      ConvertSavedModelV1ToMlir(bundle, absl::Span<std::string>(exported_names),
                                &context, import_options);
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  // Run the tf standard pipeline by default and then, run passes that lift
  // variables if the flag is set on the module.
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(module_or).value();
  mlir::PassManager pm(&context);
  std::string error;
  toolchain::raw_string_ostream error_stream(error);

  mlir::TF::StandardPipelineOptions tf_options;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

std::string ExperimentalRunPassPipeline(const std::string& mlir_txt,
                                        const std::string& pass_pipeline,
                                        bool show_debug_info,
                                        TF_Status* status) {
  RegisterPasses();
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_txt, &context);
    if (!module) {
      tsl::Set_TF_Status_from_Status(status,
                                     diagnostic_handler.ConsumeStatus());
      return "// error";
    }
  }

  // Run the pass_pipeline on the module.
  mlir::PassManager pm(&context);
  std::string error;
  toolchain::raw_string_ostream error_stream(error);
  if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Invalid pass_pipeline: " + error_stream.str()).c_str());
    return "// error";
  }

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

void ExperimentalWriteBytecode(const std::string& filename,
                               const std::string& mlir_txt, TF_Status* status) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  {
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_txt, &context);
    if (!module) {
      tsl::Set_TF_Status_from_Status(status,
                                     diagnostic_handler.ConsumeStatus());
      return;
    }
  }
  mlir::FallbackAsmResourceMap fallback_resource_map;
  mlir::BytecodeWriterConfig writer_config(fallback_resource_map);
  // TODO(jpienaar): Make this an option to the call.
  writer_config.setDesiredBytecodeVersion(1);
  std::string error;
  std::unique_ptr<toolchain::ToolOutputFile> outputFile =
      mlir::openOutputFile(filename, &error);
  if (!error.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Unable to create output file " + error).c_str());
    return;
  }
  outputFile->keep();
  if (failed(mlir::writeBytecodeToFile(*module, outputFile->os(),
                                       writer_config))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
  }
}

}  // namespace machina
