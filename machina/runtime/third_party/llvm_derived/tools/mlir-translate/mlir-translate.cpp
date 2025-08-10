//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://toolchain.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR for
// TFRT's dialect. It doesn't register any other dialects, in order to speed
// up build time. This utility is derived from MLIR's Translate Driver.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "toolchain/Support/InitLLVM.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

static toolchain::cl::opt<std::string> inputFilename(toolchain::cl::Positional,
                                                toolchain::cl::desc("<input file>"),
                                                toolchain::cl::init("-"));

static toolchain::cl::opt<std::string> outputFilename(
    "o", toolchain::cl::desc("Output filename"), toolchain::cl::value_desc("filename"),
    toolchain::cl::init("-"));

static toolchain::cl::opt<bool> splitInputFile(
    "split-input-file",
    toolchain::cl::desc("Split the input file into pieces and "
                   "process each chunk independently"),
    toolchain::cl::init(false));

static toolchain::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    toolchain::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    toolchain::cl::init(false));

int main(int argc, char **argv) {
  toolchain::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  toolchain::cl::opt<const Translation *, false, TranslationParser>
      translationRequested("", toolchain::cl::desc("Translation to perform"),
                           toolchain::cl::Required);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  toolchain::cl::ParseCommandLineOptions(argc, argv, "MLIR translation driver\n");

  std::string errorMessage;
  std::unique_ptr<toolchain::MemoryBuffer> input;
  if (auto inputAlignment = translationRequested->getInputAlignment())
    input = openInputFile(inputFilename, *inputAlignment, &errorMessage);
  else
    input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    toolchain::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    toolchain::errs() << errorMessage << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<toolchain::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;
    context.allowUnregisteredDialects();
    context.printOpOnDiagnostic(!verifyDiagnostics);
    auto sourceMgr = std::make_shared<toolchain::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), toolchain::SMLoc());

    if (!verifyDiagnostics) {
      SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
      return (*translationRequested)(sourceMgr, os, &context);
    }

    // In the diagnostic verification flow, we ignore whether the translation
    // failed (in most cases, it is expected to fail). Instead, we check if the
    // diagnostics were produced as expected.
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(*sourceMgr, &context);
    (void)(*translationRequested)(sourceMgr, os, &context);
    return sourceMgrHandler.verify();
  };

  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                     output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os()))) return 1;
  }

  output->keep();
  return 0;
}
