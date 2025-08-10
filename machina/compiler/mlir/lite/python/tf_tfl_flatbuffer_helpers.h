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
#ifndef MACHINA_COMPILER_MLIR_LITE_PYTHON_TF_TFL_FLATBUFFER_HELPERS_H_
#define MACHINA_COMPILER_MLIR_LITE_PYTHON_TF_TFL_FLATBUFFER_HELPERS_H_

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/common/tfl_pass_config.h"
#include "machina/compiler/mlir/lite/converter_flags.pb.h"
#include "machina/compiler/mlir/lite/model_flags.pb.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/lite/types.pb.h"
#include "machina/compiler/mlir/quantization/machina/python/py_function_lib.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace internal {

// Register all custom ops including user specified custom ops.
absl::Status RegisterAllCustomOps(
    const tflite::ConverterFlags& converter_flags);

// Populate quantization specs (or not) given user specified ranges for each
// input arrays.
absl::Status PopulateQuantizationSpecs(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags,
    mlir::TFL::QuantizationSpecs* quant_specs, std::vector<string>* node_names,
    std::vector<string>* node_dtypes,
    std::vector<std::optional<std::vector<int>>>* node_shapes,
    std::vector<std::optional<double>>* node_mins,
    std::vector<std::optional<double>>* node_maxs);

// Convert imported MLIR file to TfLite flatbuffer.
// This will also run relevant passes as well.
absl::Status ConvertMLIRToTFLiteFlatBuffer(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags,
    std::unique_ptr<mlir::MLIRContext>&& context,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags, string* result,
    const quantization::PyFunctionLibrary* quantization_py_function_lib);

// Give a warning for any unused flags that have been specified.
void WarningUnusedFlags(const tflite::ModelFlags& model_flags,
                        const tflite::ConverterFlags& converter_flags);
}  // namespace internal
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_LITE_PYTHON_TF_TFL_FLATBUFFER_HELPERS_H_
