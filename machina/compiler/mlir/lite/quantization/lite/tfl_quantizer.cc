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

#include <iostream>
#include <string>
#include <system_error>

#include "absl/status/status.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/InitLLVM.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/PrettyStackTrace.h"
#include "toolchain/Support/raw_ostream.h"
#include "machina/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "machina/compiler/mlir/lite/schema/schema_generated.h"

using toolchain::cl::opt;

// NOLINTNEXTLINE
static opt<std::string> inputFileName(toolchain::cl::Positional,
                                      toolchain::cl::desc("<input file>"),
                                      toolchain::cl::init("-"));

namespace mlir {
namespace {

absl::Status QuantizeAnnotatedModel(toolchain::StringRef buffer,
                                    std::string& output_buffer) {
  return mlir::lite::QuantizeModel(
      buffer, tflite::TensorType_INT8, tflite::TensorType_INT8,
      tflite::TensorType_INT8, {}, /*disable_per_channel=*/false,
      /*fully_quantize=*/true, output_buffer);
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  toolchain::InitLLVM y(argc, argv);
  toolchain::cl::ParseCommandLineOptions(argc, argv);
  auto file_or_err = toolchain::MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    toolchain::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }
  auto buffer = file_or_err->get();
  std::string output_buffer;
  if (auto status = mlir::QuantizeAnnotatedModel(buffer->getBuffer().str(),
                                                 output_buffer);
      !status.ok()) {
    toolchain::errs() << status.message() << "\n";
    return 1;
  }

  std::cout << output_buffer << "\n";
  return 0;
}
