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

#ifndef MACHINA_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
#define MACHINA_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/cc/saved_model/loader.h"
#include "machina/compiler/mlir/lite/common/tfl_pass_config.h"
#include "machina/compiler/mlir/lite/converter_flags.pb.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "machina/compiler/mlir/quantization/machina/python/py_function_lib.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/core/platform/status.h"

namespace machina {

// Load a TF model from a GraphDef definition or a TF control flow dialect MLIR
// source into a MLIR module. If `input_mlir` is true, load from a MLIR source
// file; otherwise, load from a GraphDef.
// Setting prune_unused_nodes to true, would prune unreachable nodes if
// output_arrays is specified.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    const GraphImportConfig& specs, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, toolchain::SourceMgr* source_mgr,
    mlir::MLIRContext* context);

// Load Saved model (either v1 or v2) into MLIR.
// 'saved_model_bundle' will be initialized if V1 model was loaded.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<machina::SavedModelBundle>* saved_model_bundle);

// Taking a MLIR module in TF executor dialect and a set of parameters,
// applies a set of passes (configured accordingly to the provided
// `pass_config`) to convert the module to TF Lite dialect and serializes the
// result to a string. Depending on an attribute in the module main function,
// full integer quantization is applied.
// * `quantizated_buffer_type` can be set to INT8 or FLOAT16 to trigger the
// corresponding weight quantization.
// * `export_to_mlir` enables exporting to MLIR text format, otherwise exported
// in flat buffer. If the
// * `session` pointer may provided, it will be used to freeze resource
// variables. If the `saved_model_dir` directory path is provided, then the
// `tf_saved_model.asset` ops will be freezed.
absl::Status ConvertTFExecutorToTFLOrFlatbuffer(
    std::unique_ptr<mlir::MLIRContext>&& context,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    toolchain::StringRef saved_model_dir, std::string* result, bool export_to_mlir,
    const quantization::PyFunctionLibrary* quantization_py_function_lib =
        nullptr);
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
