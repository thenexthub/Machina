/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina/xla/codegen/mlir_kernel_source.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "machina/xla/util.h"

namespace xla {

absl::StatusOr<MlirKernelSource> MlirKernelSource::ParseFromString(
    absl::string_view ir, std::unique_ptr<mlir::MLIRContext> context) {
  toolchain::SourceMgr source_mgr;

  std::string error_string;
  toolchain::raw_string_ostream error_stream(error_string);
  mlir::SourceMgrDiagnosticHandler source_mgr_handler(source_mgr, context.get(),
                                                      error_stream);

  source_mgr.AddNewSourceBuffer(toolchain::MemoryBuffer::getMemBuffer(ir),
                                toolchain::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, context.get());

  if (!mlir_module) {
    return Internal("Failed to parse MLIR IR: %s", error_string);
  }

  return MlirKernelSource(std::move(context), std::move(mlir_module));
}

}  // namespace xla
