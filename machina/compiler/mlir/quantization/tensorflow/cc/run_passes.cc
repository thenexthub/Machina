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
#include "machina/compiler/mlir/quantization/machina/cc/run_passes.h"

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/debugging/mlir_dump.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/xla/tsl/platform/errors.h"

namespace machina {
namespace quantization {

absl::Status RunPassesOnModuleOp(
    std::optional<absl::string_view> mlir_dump_file_name,
    mlir::PassManager& pass_manager, mlir::ModuleOp module_op) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module_op.getContext(),
                                                    /*propagate=*/true);

  absl::StatusOr<std::unique_ptr<toolchain::raw_ostream>> dump_file;
  if (mlir_dump_file_name) {
    TF_RETURN_IF_ERROR(machina::quantization::MaybeEnableIrPrinting(
        pass_manager, mlir_dump_file_name.value()));
  }

  if (failed(pass_manager.run(module_op))) {
    return statusHandler.ConsumeStatus();
  }

  return absl::OkStatus();
}

}  // namespace quantization
}  // namespace machina
