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

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/Path.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "machina/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "machina/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"
#include "machina/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "machina/compiler/mlir/lite/experimental/tac/tac_module.h"
#include "machina/compiler/mlir/lite/experimental/tac/tflite_import_export.h"
#include "machina/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "machina/compiler/mlir/lite/experimental/tac/utils/utils.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/core/platform/init_main.h"

using toolchain::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_file_name(toolchain::cl::Positional,
                                 toolchain::cl::desc("<input file>"),
                                 toolchain::cl::init("-"));

// NOLINTNEXTLINE
opt<std::string> output_file_name("o", toolchain::cl::desc("<output file>"),
                                  toolchain::cl::value_desc("filename"),
                                  toolchain::cl::init("-"));
// NOLINTNEXTLINE
opt<bool> input_mlir("input-mlir",
                     toolchain::cl::desc("Input is MLIR rather than FlatBuffer"),
                     toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> output_mlir(
    "output-mlir",
    toolchain::cl::desc(
        "Output MLIR rather than FlatBuffer for the generated TFLite model"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> inline_subgraphs(
    "inline-subgraphs",
    toolchain::cl::desc("Whether or not to inline all the subgraphs"),
    toolchain::cl::init(true));

// NOLINTNEXTLINE
opt<std::string> device_specs(
    "device-specs", toolchain::cl::desc("comma separated list of device specs."),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<bool> export_runtime_metadata(
    "export-runtime-metadata",
    toolchain::cl::desc("Whether or not to export metadata, if yes, the metadata "
                   "file will be exported along with the output model"),
    toolchain::cl::init(false));

namespace {

std::unique_ptr<mlir::TFL::tac::TacImporter> CreateTfLiteImporter() {
  mlir::TFL::tac::TfLiteImporter::Options options;
  options.file_name = input_file_name;
  options.input_mlir = input_mlir;
  return std::make_unique<mlir::TFL::tac::TfLiteImporter>(options);
}

std::unique_ptr<mlir::TFL::tac::TacExporter> CreateTfLiteExporter(
    const std::vector<std::string>& target_hardware_backends) {
  mlir::TFL::tac::TfLiteExporter::Options options;
  options.output_mlir = output_mlir;
  options.output_file_name = output_file_name;
  options.export_runtime_metadata = export_runtime_metadata;
  options.target_hardware_backends = target_hardware_backends;
  return std::make_unique<mlir::TFL::tac::TfLiteExporter>(options);
}

absl::Status TargetAwareConversionMain() {
  std::vector<std::string> device_specs_array =
      absl::StrSplit(device_specs, ',', absl::SkipEmpty());
  mlir::TFL::tac::TacModule::Options options;
  options.hardware_backends = device_specs_array;
  options.debug_mode = true;
  if (!output_mlir || inline_subgraphs) {
    options.debug_mode = false;
  }
  options.enable_inliner = true;
  options.legalize_to_tflite_ops = true;
  mlir::TFL::tac::TacModule tac_module(options);
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  tac_module.RegisterExtraDialects(registry);
  tac_module.SetImporter(CreateTfLiteImporter());
  tac_module.SetExporter(CreateTfLiteExporter(options.hardware_backends));
  return tac_module.Run();
}
}  // namespace

int main(int argc, char** argv) {
  machina::InitMlir y(&argc, &argv);

  toolchain::cl::ParseCommandLineOptions(argc, argv, "Target aware conversion\n");

  absl::Status status = TargetAwareConversionMain();
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return 0;
}
