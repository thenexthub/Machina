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

#include "machina/compiler/mlir/machina/utils/error_util.h"

#include <string_view>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/util/managed_stack_trace.h"

namespace mlir {

StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    MLIRContext* context, bool propagate, bool filter_stack)
    : BaseScopedDiagnosticHandler(context, propagate, filter_stack) {
  if (filter_stack) {
    this->shouldShowLocFn = [](Location loc) -> bool {
      // For a Location to be surfaced in the stack, it must evaluate to true.
      // For any Location that is a FileLineColLoc:
      if (FileLineColLoc fileLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
        return !machina::IsInternalFrameForFilename(
            fileLoc.getFilename().str());
      } else {
        // If this is a non-FileLineColLoc, go ahead and include it.
        return true;
      }
    };
  }

  setHandler([this](Diagnostic& diag) { return this->handler(&diag); });
}

Status StatusScopedDiagnosticHandler::ConsumeStatus() {
  return BaseScopedDiagnosticHandler::ConsumeStatus();
}

Status StatusScopedDiagnosticHandler::Combine(Status status) {
  absl::Status absl_s = BaseScopedDiagnosticHandler::Combine(status);

  return absl_s;
}

}  // namespace mlir
