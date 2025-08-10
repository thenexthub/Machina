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

#include "machina/compiler/mlir/lite/quantization/lite/quantize_model.h"

#include <optional>
#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/Twine.h"
#include "toolchain/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/common/tfl_pass_config.h"
#include "machina/compiler/mlir/lite/debug/debug.h"
#include "machina/compiler/mlir/lite/debug/debug_options.pb.h"
#include "machina/compiler/mlir/lite/flatbuffer_export.h"
#include "machina/compiler/mlir/lite/flatbuffer_import.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "machina/compiler/mlir/lite/schema/schema_generated.h"
#include "machina/compiler/mlir/lite/tf_tfl_passes.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/lite/utils/convert_type.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/core/framework/types.pb.h"

namespace mlir {
namespace lite {

std::string TfLiteToMlir(const absl::string_view tflite_op_name) {
  toolchain::StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return toolchain::Twine("tfl.", op_name.lower()).str();
}

// TODO(fengliuai): check the result for `fully_quantize` flag.
absl::Status QuantizeModel(
    const absl::string_view model_buffer, const tflite::TensorType &input_type,
    const tflite::TensorType &output_type,
    const tflite::TensorType &inference_type,
    const std::unordered_set<std::string> &operator_names,
    bool disable_per_channel, bool fully_quantize, std::string &output_buffer,
    bool verify_numeric, bool whole_model_verify, bool legacy_float_scale,
    const absl::flat_hash_set<std::string> &denylisted_ops,
    const absl::flat_hash_set<std::string> &denylisted_nodes,
    const bool enable_variable_quantization,
    bool disable_per_channel_for_dense_layers,
    const std::optional<const machina::converter::DebugOptions>
        &debug_options) {
  // Translate TFLite names to mlir op names.
  absl::flat_hash_set<std::string> denylisted_mlir_op_names;
  for (const auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      model_buffer, &context, UnknownLoc::get(&context));
  if (!module) {
    return absl::InternalError("Couldn't import flatbuffer to MLIR.");
  }

  // Apply quantization passes.
  PassManager pm((*module)->getName(), OpPassManager::Nesting::Implicit);
  if (debug_options.has_value()) {
    // Add debugging instrumentation
    machina::InitPassManager(pm, debug_options.value());
  }
  TFL::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.post_training_quantization = true;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.disable_per_channel_for_dense_layers =
      disable_per_channel_for_dense_layers;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = denylisted_mlir_op_names;
  quant_specs.nodes_blocklist = denylisted_nodes;
  quant_specs.enable_mlir_variable_quantization = enable_variable_quantization;

  toolchain::dbgs() << "fully_quantize: " << fully_quantize
               << ", inference_type: " << quant_specs.inference_type
               << ", input_inference_type: "
               << tflite::EnumNameTensorType(input_type)
               << ", output_inference_type: "
               << tflite::EnumNameTensorType(output_type) << "\n";
  mlir::Builder mlir_builder(&context);
  mlir::Type input_mlir_type =
      tflite::ConvertElementType(input_type, mlir_builder);
  mlir::Type output_mlir_type =
      tflite::ConvertElementType(output_type, mlir_builder);

  if (fully_quantize) {
    input_mlir_type = tflite::ConvertElementType(inference_type, mlir_builder);
    output_mlir_type = input_mlir_type;
  }

  machina::AddQuantizationPasses(mlir::TFL::PassConfig(quant_specs), pm);
  pm.addPass(TFL::CreateModifyIONodesPass(input_mlir_type, output_mlir_type));
  // If the first or final ops are not quantized, remove QDQ.
  pm.addPass(TFL::CreatePostQuantizeRemoveQDQPass());
  if (failed(pm.run(module.get()))) {
    const std::string err(statusHandler.ConsumeStatus().message());
    return absl::InternalError(err);
  }

  // Export the results.
  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_force_select_tf_ops(false);
  options.converter_flags.set_enable_select_tf_ops(true);
  options.converter_flags.set_allow_custom_ops(true);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &output_buffer)) {
    return absl::InternalError("Failed to export MLIR to flatbuffer.");
  }
  return absl::OkStatus();
}

}  // namespace lite
}  // namespace mlir
