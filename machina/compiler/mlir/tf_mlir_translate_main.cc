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

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "toolchain/Support/InitLLVM.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/ToolUtilities.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/machina/translate/tf_mlir_translate.h"
#include "machina/compiler/mlir/tools/tf_mlir_translate_cl.h"
#include "machina/core/platform/init_main.h"

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> input_filename(toolchain::cl::Positional,
                                                 toolchain::cl::desc("<input file>"),
                                                 toolchain::cl::init("-"));

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> output_filename(
    "o", toolchain::cl::desc("Output filename"), toolchain::cl::value_desc("filename"),
    toolchain::cl::init("-"));

// NOLINTNEXTLINE
static toolchain::cl::opt<bool> splitInputFile(
    "split-input-file",
    toolchain::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
static toolchain::cl::opt<bool> import_saved_model_object_graph(
    "savedmodel-objectgraph-to-mlir",
    toolchain::cl::desc(
        "Import a saved model's object graph to its MLIR representation"),
    toolchain::cl::value_desc("dir"));

// NOLINTNEXTLINE
static toolchain::cl::opt<bool> import_saved_model_signature_defs(
    "savedmodel-signaturedefs-to-mlir",
    toolchain::cl::desc(
        "Import a saved model's SignatureDefs to their MLIR representation"),
    toolchain::cl::value_desc("dir"));

// NOLINTNEXTLINE
static toolchain::cl::opt<bool> import_saved_model_signature_defs_lite(
    "savedmodel-signaturedefs-to-mlir-lite",
    toolchain::cl::desc("Import a saved model's SignatureDefs to their MLIR "
                   "representation without any graph transformation"),
    toolchain::cl::value_desc("dir"));

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    toolchain::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    toolchain::cl::init("serve"));

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    toolchain::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    toolchain::cl::init(""));

int main(int argc, char** argv) {
  machina::InitMlir y(&argc, &argv);

  // Add flags for all the registered translations.
  toolchain::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>
      requested_translation("", toolchain::cl::desc("Translation to perform"));
  mlir::registerAsmPrinterCLOptions();
  toolchain::cl::ParseCommandLineOptions(argc, argv, "TF MLIR translation driver\n");

  if (!import_saved_model_object_graph && !import_saved_model_signature_defs &&
      !import_saved_model_signature_defs_lite && !requested_translation) {
    toolchain::errs() << "error: need to specify one translation to perform\n";
    return 1;
  } else if (import_saved_model_object_graph +
                 import_saved_model_signature_defs +
                 import_saved_model_signature_defs_lite +
                 (requested_translation != nullptr) >
             1) {
    toolchain::errs()
        << "error: cannot specify more than one translation to perform\n";
    return 1;
  }

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    toolchain::errs() << error_message << "\n";
    return 1;
  }

  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  std::vector<std::string> exported_names_vector =
      absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
  absl::Span<std::string> exported_names(exported_names_vector);

  if (import_saved_model_object_graph) {
    mlir::MLIRContext context;
    auto module_or = machina::SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, &context);
    if (!module_or.status().ok()) return 1;

    module_or.value()->print(output->os());
  } else if (import_saved_model_signature_defs) {
    mlir::MLIRContext context;
    machina::MLIRImportOptions import_options;
    import_options.upgrade_legacy = upgrade_legacy;
    auto module_or = machina::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, &context, import_options);
    if (!module_or.status().ok()) return 1;

    module_or.value()->print(output->os());
  } else if (import_saved_model_signature_defs_lite) {
    mlir::MLIRContext context;
    machina::MLIRImportOptions import_options;
    import_options.upgrade_legacy = upgrade_legacy;
    auto module_or = machina::SavedModelSignatureDefsToMlirImportLite(
        input_filename, tags, exported_names, &context, import_options);
    if (!module_or.status().ok()) return 1;

    module_or.value()->print(output->os());
  } else {
    auto input = mlir::openInputFile(input_filename, &error_message);

    if (!input) {
      toolchain::errs() << error_message << "\n";
      return 1;
    }

    // Processes the memory buffer with a new MLIRContext.
    auto processBuffer = [&](std::unique_ptr<toolchain::MemoryBuffer> ownedBuffer,
                             toolchain::raw_ostream& os) {
      auto sourceMgr = std::make_shared<toolchain::SourceMgr>();
      sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), toolchain::SMLoc());
      mlir::MLIRContext context;
      mlir::SourceMgrDiagnosticHandler diagnostic_handler(*sourceMgr, &context);
      return (*requested_translation)(sourceMgr, os, &context);
    };

    if (splitInputFile) {
      if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                             output->os())))
        return 1;
    } else {
      if (failed(processBuffer(std::move(input), output->os()))) return 1;
    }
  }

  output->keep();
  return 0;
}
