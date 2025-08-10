/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/function/function.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "machina/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "machina/core/framework/function.h"
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace machina {

struct FunctionBody;

// Converts an MLIR `module` in TF dialect to TFRT's Binary Executable Format.
// If `fallback_state` is not null, the MLIR functions for XLA clusters in
// the form of XlaLaunch will be exported and added to the function library when
// needed. The nested functions will also be exported. If
// `added_xla_function_names` is not null, it will be populated with the names
// of the added XLA functions.
absl::Status ConvertTfMlirToBef(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    tfrt::BefBuffer* bef_buffer, tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state = nullptr,
    std::vector<std::string>* added_xla_function_names = nullptr);

absl::Status ConvertTfMlirToRuntimeExecutable(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    absl::FunctionRef<
        absl::Status(mlir::PassManager&, mlir::ModuleOp,
                     const machina::TfrtPipelineOptions& options)>
        emit_executable,
    tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state = nullptr,
    std::vector<std::string>* added_xla_function_names = nullptr);

std::unique_ptr<machina::TfrtPipelineOptions> GetTfrtPipelineOptions(
    const TfrtCompileOptions& options);

// Adds MLIR functions for XLA clusters to the function library.
absl::Status AddXlaFunctions(
    tfrt_stub::FallbackState* fallback_state, mlir::ModuleOp mlir_module,
    std::vector<std::string>* added_xla_function_names = nullptr);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_
