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
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/flatbuffer_import.h"

namespace machina {
namespace {
static mlir::OwningOpRef<mlir::ModuleOp> FlatBufferFileToMlirTranslation(
    toolchain::SourceMgr* source_mgr, mlir::MLIRContext* context) {
  const toolchain::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc =
      mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, false, inputs, outputs, false);
}

}  // namespace

std::string FlatBufferFileToMlir(const std::string& model_file_or_buffer,
                                 bool input_is_filepath) {
  // referred logic from mlir::mlirTranslateMain().

  std::string errorMessage;
  std::unique_ptr<toolchain::MemoryBuffer> input;
  if (input_is_filepath) {
    input = mlir::openInputFile(model_file_or_buffer, &errorMessage);
    if (!input) {
      toolchain::errs() << errorMessage << "\n";
      return "";
    }
  } else {
    input = toolchain::MemoryBuffer::getMemBuffer(model_file_or_buffer, "flatbuffer",
                                             false);
    if (!input) {
      toolchain::errs() << "Can't get toolchain::MemoryBuffer\n";
      return "";
    }
  }

  mlir::MLIRContext context;
  context.printOpOnDiagnostic(true);
  toolchain::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), toolchain::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      FlatBufferFileToMlirTranslation(&sourceMgr, &context);
  if (!module || failed(verify(*module))) return "";

  std::string mlir_output;
  toolchain::raw_string_ostream output_stream(mlir_output);
  // Dump MLIR with eliding large elements.
  module->print(
      output_stream,
      mlir::OpPrintingFlags().useLocalScope().elideLargeElementsAttrs());
  return mlir_output;
}

}  // namespace machina
