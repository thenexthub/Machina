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
#include "machina/compiler/mlir/tfrt/utils/export.h"

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"
#include "machina/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "machina/core/framework/function.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/traceme.h"

namespace machina {

absl::Status ExportFunctionDefs(
    mlir::ModuleOp module,
    absl::AnyInvocable<absl::Status(machina::FunctionDef)> callback,
    bool export_tf_original_func_name) {
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode(
        "ExportFunctionDefs",
        {{"module_name", absl::string_view(module.getName().value_or("?"))}});
  });

  TF_RETURN_IF_ERROR(
      machina::tf2xla::v1::ExportFromTensorflowDialectToExecutor(module));

  {
    mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::CreateBreakUpIslandsPass());

    if (mlir::failed(pm.run(module))) {
      return diag_handler.ConsumeStatus();
    }
  }
  machina::GraphExportConfig configs;
  configs.export_original_tf_func_name = export_tf_original_func_name;

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    machina::FunctionDef function_def;
    TF_RETURN_IF_ERROR(
        machina::tf2xla::v2::ConvertMlirFunctionToFunctionLibraryDef(
            func, configs, &function_def));
    TF_RETURN_IF_ERROR(callback(std::move(function_def)));
  }

  return absl::OkStatus();
}

}  // namespace machina
