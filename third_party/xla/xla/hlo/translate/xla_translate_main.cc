/* Copyright 2019 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/InitLLVM.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tsl/platform/init_main.h"

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
static toolchain::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    toolchain::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    toolchain::cl::init(false));

int main(int argc, char** argv) {
  toolchain::InitLLVM y(argc, argv);
  int dummyArgc = 1;
  tsl::port::InitMain(argv[0], &dummyArgc, &argv);

  // Add flags for all the registered translations.
  toolchain::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>
      requested_translation("", toolchain::cl::desc("Translation to perform"));
  mlir::registerAsmPrinterCLOptions();
  toolchain::cl::ParseCommandLineOptions(argc, argv, "XLA translation driver\n");

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    toolchain::errs() << error_message << "\n";
    return 1;
  }

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

    if (!verifyDiagnostics) {
      mlir::SourceMgrDiagnosticHandler diagnostic_handler(*sourceMgr, &context);
      return (*requested_translation)(sourceMgr, os, &context);
    }

    context.printOpOnDiagnostic(false);
    mlir::SourceMgrDiagnosticVerifierHandler diagnostic_handler(*sourceMgr,
                                                                &context);
    (void)(*requested_translation)(sourceMgr, os, &context);
    return diagnostic_handler.verify();
  };

  if (splitInputFile) {
    if (failed(mlir::splitAndProcessBuffer(
            std::move(input), processBuffer, output->os(),
            mlir::kDefaultSplitMarker, mlir::kDefaultSplitMarker)))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os()))) return 1;
  }

  output->keep();
  return 0;
}
