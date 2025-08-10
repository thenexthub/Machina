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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"
#include "machina/compiler/mlir/quantization/machina/quantize_preprocess.h"
#include "machina/compiler/mlir/machina/transforms/shape_inference.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/public/session.h"

namespace mlir {
namespace {

// Extract the mlir TF module and optionally a ::machina::SavedModelBundle
// from a saved model or from an mlir file.
absl::StatusOr<quant::stablehlo::ImportedMlirModuleOp> ImportSavedModelOrTfMlir(
    absl::string_view input_path, MLIRContext* context,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names, bool is_input_mlir_module) {
  if (is_input_mlir_module) {
    std::string error_message;
    std::unique_ptr<toolchain::MemoryBuffer> file =
        openInputFile(input_path, &error_message);
    if (!file) {
      return absl::AbortedError(
          absl::StrCat("Failed to parse input MLIR model: ", error_message));
    }

    toolchain::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(file), toolchain::SMLoc());
    auto module = parseSourceFile<ModuleOp>(source_mgr, context);
    if (module->getOperation() == nullptr) {
      return absl::AbortedError("Failed to parse input MLIR model.");
    }

    return quant::stablehlo::ImportedMlirModuleOp(std::move(module), nullptr);
  }

  std::unordered_set<std::string> tag_set(tag_names.begin(), tag_names.end());
  return quant::stablehlo::SavedModelToMlirModuleOp(
      input_path, tag_set, exported_model_signatures, *context);
}

// Convert an TF module to a StableHLO module
absl::StatusOr<OwningOpRef<ModuleOp>> ConvertTFToStablehlo(
    quant::stablehlo::ImportedMlirModuleOp imported_module,
    absl::string_view input_path, MLIRContext* context,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str, bool is_input_mlir_module) {
  auto [module_op, saved_model_bundle] = std::move(imported_module);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  if (!is_input_mlir_module) {
    std::unordered_set<std::string> tag_set(tag_names.begin(), tag_names.end());
    TF_ASSIGN_OR_RETURN(
        auto function_aliases,
        quant::stablehlo::GetFunctionAliases(input_path, tag_set));
    quant::stablehlo::UpdateFunctionAliases(function_aliases, *module_op);
    absl::c_for_each(function_aliases, [&](const auto& aliases) {
      return aliased_function_names.insert(aliases.first);
    });
  }

  std::optional<machina::Session*> session;
  if (saved_model_bundle) {
    session = saved_model_bundle->GetSession();
  }
  TF_ASSIGN_OR_RETURN(auto input_arg_shapes_vec,
                      TF::ParseArgumentShapes(input_arg_shapes_str));
  toolchain::SmallVector<toolchain::ArrayRef<int64_t>> input_arg_shapes(
      input_arg_shapes_vec.begin(), input_arg_shapes_vec.end());
  TF_RETURN_IF_ERROR(machina::quantization::PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/"", /*is_inliner_run=*/true,
      /*noinline_functions=*/aliased_function_names, *module_op, context,
      session,
      /*run_tf_to_stablehlo=*/true, /*deserialize_xla_call_module=*/false,
      input_arg_shapes));

  return std::move(module_op);
}

}  // namespace

absl::StatusOr<OwningOpRef<ModuleOp>> TfToStablehlo(
    absl::string_view input_path, MLIRContext* context,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str, bool is_input_mlir_module) {
  auto import_module_status =
      ImportSavedModelOrTfMlir(input_path, context, exported_model_signatures,
                               tag_names, is_input_mlir_module);
  if (!import_module_status.ok()) {
    return import_module_status.status();
  }

  return ConvertTFToStablehlo(*std::move(import_module_status), input_path,
                              context, tag_names, input_arg_shapes_str,
                              is_input_mlir_module);
}

}  // namespace mlir
