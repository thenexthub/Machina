/* Copyright 2020 The OpenXLA Authors.

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

#include "toolchain/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "machina/xla/hlo/translate/hlo_to_mhlo/translate.h"

namespace {
// NOLINTNEXTLINE
toolchain::cl::opt<bool> import_all_computations(
    "hlo-import-all-computations",
    toolchain::cl::desc("Enable importing unreachable computations."));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> flatten_computation_args_result(
    "hlo-flatten-computation-args-result",
    toolchain::cl::desc("Enable flattening computation arguments and results."));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> emit_stablehlo(
    "emit-stablehlo",
    toolchain::cl::desc("Allow a mix of MHLO and StableHLO ops in the output."));
}  // namespace

static mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslate(
    toolchain::StringRef input, mlir::MLIRContext* context) {
  return xla::HloToMlirHloTranslateFunction(
      input, context, import_all_computations, flatten_computation_args_result,
      emit_stablehlo);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslate(
    toolchain::StringRef input, mlir::MLIRContext* context) {
  return xla::HloTextToMlirHloTranslateFunction(
      input, context, import_all_computations, flatten_computation_args_result,
      emit_stablehlo);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloToStablehloTranslate(
    toolchain::StringRef input, mlir::MLIRContext* context) {
  if (!flatten_computation_args_result.getValue() ||
      !import_all_computations.getValue()) {
    mlir::emitWarning(mlir::UnknownLoc::get(context),
                      "HLO => StableHLO requires flattened_args and "
                      "import_all_computations to be set to true.");
  }
  return xla::HloToStablehloTranslateFunction(input, context);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloTextToStablehloTranslate(
    toolchain::StringRef input, mlir::MLIRContext* context) {
  if (!flatten_computation_args_result.getValue() ||
      !import_all_computations.getValue()) {
    mlir::emitWarning(mlir::UnknownLoc::get(context),
                      "HLO => StableHLO requires flattened_args and "
                      "import_all_computations to be set to true.");
  }
  return xla::HloTextToStablehloTranslateFunction(input, context);
}

static mlir::TranslateToMLIRRegistration HloToMlirHloTranslateRegistration(
    "hlo-to-mlir-hlo", "hlo-to-mlir-hlo", HloToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloTextToMlirHloTranslateRegistration(
    "hlo-text-to-mlir-hlo", "hlo-text-to-mlir-hlo", HloTextToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloToStablehloTranslateRegistration(
    "hlo-to-stablehlo", "hlo-to-stablehlo", HloToStablehloTranslate);

static mlir::TranslateToMLIRRegistration
    HloTextToStablehloTranslateRegistration("hlo-text-to-stablehlo",
                                            "hlo-text-to-stablehlo",
                                            HloTextToStablehloTranslate);
