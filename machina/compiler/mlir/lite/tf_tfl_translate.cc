/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/lite/common/tfl_pass_config.h"
#include "machina/compiler/mlir/lite/converter_flags.pb.h"
#include "machina/compiler/mlir/lite/flatbuffer_export_flags.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "machina/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "machina/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/xla/hlo/translate/hlo_to_mhlo/translate.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/errors.h"

using toolchain::cl::opt;
using mlir::MLIRContext;
using mlir::ModuleOp;

// NOLINTNEXTLINE
opt<bool> upgrade_legacy("tf-upgrade-legacy",
                         toolchain::cl::desc("Upgrade legacy TF graph behavior"),
                         toolchain::cl::init(false));

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> weight_quantization(
    "weight_quantization",
    toolchain::cl::desc("The type of the quantized weight buffer. Must be NONE, "
                   "INT8, FLOAT16."),
    toolchain::cl::init("NONE"));

enum TranslationStatus { kTrSuccess, kTrFailure };

int main(int argc, char **argv) {
  // TODO(jpienaar): Revise the command line option parsing here.
  machina::InitMlir y(&argc, &argv);

  // TODO(antiagainst): We are pulling in multiple transformations as follows.
  // Each transformation has its own set of command-line options; options of one
  // transformation can essentially be aliases to another. For example, the
  // -tfl-annotate-inputs has -tfl-input-arrays, -tfl-input-data-types, and
  // -tfl-input-shapes, which are the same as -graphdef-to-mlir transformation's
  // -tf_input_arrays, -tf_input_data_types, and -tf_input_shapes, respectively.
  // We need to disable duplicated ones to provide a cleaner command-line option
  // interface. That also means we need to relay the value set in one option to
  // all its aliases.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  toolchain::cl::ParseCommandLineOptions(
      argc, argv, "TF GraphDef to TFLite FlatBuffer converter\n");

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);

  if (input_mlir) {
    // TODO(@zichuanwei): hack to enable mlir conversion via this tool, will get
    // back to do it properly in the future
    mlir::DialectRegistry registry;
    RegisterAllTensorFlowDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                    mlir::TFL::TensorFlowLiteDialect, mlir::mhlo::MhloDialect,
                    mlir::chlo::ChloDialect>();
    context->appendDialectRegistry(registry);
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module;
  std::unordered_set<std::string> tags;

  machina::GraphImportConfig specs;
  specs.upgrade_legacy = upgrade_legacy;
  specs.prune_unused_nodes = true;

  if (!select_user_tf_ops.empty() && !emit_select_tf_ops) {
    toolchain::errs() << "You must specify `emit-select-tf-ops=true` when passing "
                    "`select-user-tf-ops` flag.";
    return kTrFailure;
  }

  // TODO(b/147435528): We need to test the e2e behavior once the graph freezing
  // inside mlir is done.
  if ((import_saved_model_object_graph || import_saved_model_signature_defs) &&
      import_hlo) {
    toolchain::errs() << "Import saved model and import hlo cannot be both set.";
    return kTrFailure;
  }

  if (import_saved_model_object_graph || import_saved_model_signature_defs) {
    // Saved model import path.
    int saved_model_version;
    if (import_saved_model_object_graph) {
      saved_model_version = 2;
    } else {
      saved_model_version = 1;
    }
    if (input_mlir)
      module = machina::errors::InvalidArgument(
          "Importing saved model should not have input_mlir set");

    tags = absl::StrSplit(saved_model_tags, ',');
    std::vector<std::string> exported_names_vector =
        absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
    absl::Span<std::string> exported_names(exported_names_vector);

    std::vector<std::string> extra_opdefs(custom_opdefs.begin(),
                                          custom_opdefs.end());
    module = machina::ImportSavedModel(
        input_file_name, saved_model_version, tags, extra_opdefs,
        exported_names, specs, /*enable_variable_lifting=*/true, context.get(),
        /*saved_model_bundle=*/nullptr);
  } else if (import_hlo) {
    // HLO import path.
    std::string error;
    std::unique_ptr<toolchain::MemoryBuffer> buffer =
        mlir::openInputFile(input_file_name, &error);
    if (buffer == nullptr) {
      toolchain::errs() << "Cannot open input file: " << input_file_name << " "
                   << error;
      return kTrFailure;
    }

    auto content = buffer->getBuffer();
    if (hlo_import_type == HloImportType::hlotxt) {
      module =
          xla::HloTextToMlirHloTranslateFunction(content, context.get(), false);
    } else if (hlo_import_type == HloImportType::proto) {
      module =
          xla::HloToMlirHloTranslateFunction(content, context.get(), false);
    } else {
      module = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::parseSourceString<mlir::ModuleOp>(content, context.get()));
    }
  } else {
    // Graphdef import path.
    toolchain::SourceMgr source_mgr;
    module = machina::LoadFromGraphdefOrMlirSource(
        input_file_name, input_mlir, use_splatted_constant, custom_opdefs,
        specs, debug_info_file, input_arrays, input_dtypes, input_shapes,
        output_arrays, control_output_arrays, &source_mgr, context.get());
  }

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  // Set the quantization specifications from the command line flags.
  mlir::TFL::QuantizationSpecs quant_specs;
  if (mlir::TFL::ParseInputNodeQuantSpecs(input_arrays, min_values, max_values,
                                          inference_type, &quant_specs)) {
    toolchain::errs() << "Failed to get input quant spec.";
    return kTrFailure;
  }
  if (weight_quantization != "NONE") {
    quant_specs.weight_quantization = true;
    if (weight_quantization == "INT8") {
      quant_specs.inference_type = machina::DT_QINT8;
    } else if (weight_quantization == "FLOAT16") {
      quant_specs.inference_type = machina::DT_HALF;
    } else {
      toolchain::errs() << "Unknown weight quantization " << weight_quantization;
      return kTrFailure;
    }
  }
  if (!emit_quant_adaptor_ops) {
    quant_specs.inference_input_type = quant_specs.inference_type;
  }

  if (!quant_stats_file_name.empty()) {
    std::string error_message;
    auto file = mlir::openInputFile(quant_stats_file_name, &error_message);
    if (!file) {
      toolchain::errs() << "fail to open quant stats file: "
                   << quant_stats_file_name;
      return kTrFailure;
    }
    quant_specs.serialized_quant_stats = file->getBuffer().str();
  }

  tflite::ConverterFlags::ModelOriginFramework model_origin_framework_enum;
  tflite::ConverterFlags::ModelOriginFramework_Parse(
      model_origin_framework, &model_origin_framework_enum);

  mlir::TFL::PassConfig pass_config(quant_specs);
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.lower_tensor_list_ops = lower_tensor_list_ops;
  pass_config.unfold_batch_matmul = unfold_batchmatmul;
  pass_config.unfold_large_splat_constant = unfold_large_splat_constant;
  pass_config.guarantee_all_funcs_one_use = guarantee_all_funcs_one_use;
  pass_config.enable_dynamic_update_slice = enable_dynamic_update_slice;
  pass_config.runtime_verification = true;
  pass_config.outline_tf_while = true;
  pass_config.preserve_assert_op = preserve_assert_op;
  pass_config.enable_stablehlo_conversion = enable_stablehlo_conversion;
  pass_config.legalize_custom_tensor_list_ops = legalize_custom_tensor_list_ops;
  pass_config.enable_hlo_to_tf_conversion = enable_hlo_to_tf_conversion;
  pass_config.disable_hlo_to_tfl_conversion = disable_hlo_to_tfl_conversion;
  pass_config.reduce_type_precision = reduce_type_precision;
  pass_config.model_origin_framework = model_origin_framework_enum;
  pass_config.enable_composite_direct_lowering =
      enable_composite_direct_lowering;

  tflite::ConverterFlags converter_flags;
  converter_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  converter_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  converter_flags.set_allow_custom_ops(emit_custom_ops);
  converter_flags.set_allow_all_select_tf_ops(allow_all_select_tf_ops);
  converter_flags.set_enable_dynamic_update_slice(enable_dynamic_update_slice);
  converter_flags.set_post_training_quantize(post_training_quantization);
  converter_flags.set_use_buffer_offset(use_buffer_offset);
  converter_flags.set_legalize_custom_tensor_list_ops(
      legalize_custom_tensor_list_ops);
  converter_flags.set_reduce_type_precision(reduce_type_precision);
  converter_flags.set_enable_composite_direct_lowering(
      enable_composite_direct_lowering);
  converter_flags.set_model_origin_framework(model_origin_framework_enum);

  // Read list of user select ops.
  toolchain::SmallVector<toolchain::StringRef, 2> user_ops;
  (toolchain::StringRef(select_user_tf_ops))
      .split(user_ops, ',', /*MaxSplit=*/-1,
             /*KeepEmpty=*/false);
  toolchain::for_each(user_ops, [&converter_flags](toolchain::StringRef op_name) {
    *(converter_flags.add_select_user_tf_ops()) = op_name.str();
  });

  std::string result;
  auto status = machina::ConvertTFExecutorToTFLOrFlatbuffer(
      std::move(context), std::move(module.value()), converter_flags,
      pass_config, tags, /*saved_model_dir=*/"", &result,
      /*export_to_mlir=*/output_mlir);
  if (!status.ok()) {
    toolchain::errs() << status.message() << '\n';
    return kTrFailure;
  }

  std::string error_msg;
  auto output = mlir::openOutputFile(output_file_name, &error_msg);
  if (output == nullptr) {
    toolchain::errs() << error_msg << '\n';
    return kTrFailure;
  }
  output->os() << result;
  output->keep();

  return kTrSuccess;
}
