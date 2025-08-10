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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TF_MLIR_TRANSLATE_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TF_MLIR_TRANSLATE_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/cc/saved_model/loader.h"
#include "machina/compiler/mlir/machina/translate/mlir_import_options.h"

namespace machina {

// Converts a TensorFlow SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelObjectGraphToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    bool unconditionally_use_set_output_shapes = false,
    bool import_variables_as_dense_resources = false);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
// 'saved_model_bundle' if not null, will be initialized with the model bundle.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options,
    std::unique_ptr<machina::SavedModelBundle>* saved_model_bundle =
        nullptr);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`. This does not create session internally so it is faster
// and does not perform any graph transformation.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImportLite(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TF_MLIR_TRANSLATE_H_
