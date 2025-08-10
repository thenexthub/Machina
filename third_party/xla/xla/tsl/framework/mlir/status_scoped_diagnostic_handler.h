/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_XLATSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_
#define MACHINA_XLATSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_

#include <string>

#include "absl/status/status.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

namespace tsl {

// Diagnostic handler that collects all of the diagnostics reported and produces
// an absl::Status to return to callers.
class StatusScopedDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  explicit StatusScopedDiagnosticHandler(mlir::MLIRContext* context);

  // Destruction CHECK-fails if ConsumeStatus has not been called.
  ~StatusScopedDiagnosticHandler();

  // Returns the aggregate status.
  absl::Status consumeStatus();

  // Returns the aggregate status, if it is non-OK, or an error, if `result` is
  // mlir::failed. If the aggregate status is OK and mlir::succeeded(result),
  // returns OK.
  absl::Status consumeStatus(mlir::LogicalResult result);

 private:
  mlir::LogicalResult handleDiagnostic(mlir::Diagnostic& diag);

  std::string diag_str_;
  toolchain::raw_string_ostream diag_stream_;
  toolchain::SourceMgr source_mgr_;
  absl::Status status_;
  bool consumed_ = false;
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_
