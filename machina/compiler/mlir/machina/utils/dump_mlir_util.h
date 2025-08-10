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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_MLIR_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_MLIR_UTIL_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"

namespace machina {

inline constexpr absl::string_view kCrashReproducerStdErr = "-";
inline constexpr absl::string_view kCrashReproducerCrashAnalysis =
    "crash_analysis";

// Creates a file to use for dumping and returns success if a file could be
// created. The opened file is placed in 'os' and the path of the file used is
// placed in 'filepath'.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is kCrashReproducerStdErr,
// then the LOG(INFO) macro is used instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
absl::Status CreateFileForDumping(toolchain::StringRef name,
                                  std::unique_ptr<toolchain::raw_ostream>* os,
                                  std::string* filepath,
                                  toolchain::StringRef dirname = "");

// Dumps MLIR operation to a file and returns the file name used.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is kCrashReproducerStdErr,
// then the MLIR operation will be logged (using the LOG(INFO) macro) instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
// If `pass_manager` is provided, prints a header with the pass pipeline.
std::string DumpMlirOpToFile(toolchain::StringRef name, mlir::Operation* op,
                             toolchain::StringRef dirname = "",
                             const mlir::PassManager* pass_manager = nullptr);

// Reads the directory to dump the MLIR module from environment variables.
// Default is reading from TF_DUMP_GRAPH_PREFIX, and if the string is 'sponge'
// read from TEST_UNDECLARED_OUTPUTS_DIR. Returns nullptr if the directory
// cannot be determined and generates a warning message.
std::string GetDumpDirFromEnvVar();

// Dumps a raw string to a file and returns the file name used.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
std::string DumpRawStringToFile(toolchain::StringRef name, toolchain::StringRef content,
                                toolchain::StringRef dirname = "");

// Enable the crash reproducer on the provided PassManager to the provided
// directory path.
// If the provided path is empty, it is retrieved from the
// environment variable `MLIR_CRASH_REPRODUCER_DIRECTORY`.
// If the provided path is the string "sponge", the file will be included
// in the sponge "Output Files" by looking up the environment to infer
// the directory path.
// If the provided path is the string kCrashReproducerStdErr, the data is
// dumped into the stderr.
// If the provided path is the string kCrashReproducerCrashAnalysis, the data
// is dumped to the crash analysis system. Note, environment var
// `MLIR_CRASH_REPRODUCER_DIRECTORY` can be used to override
// kCrashReproducerCrashAnalysis settings.
void SetCrashReproducer(mlir::PassManager& pm, toolchain::StringRef dir_path = "");

// This applies both the PassManagerCLOptions provided by MLIR along with any
// machina specific options.
//
// Note that this function should be in a more appropriate file, but it is
// unclear what a proper file would be as no other functions would currently be
// in the file also.
void applyTensorflowAndCLOptions(mlir::PassManager& pm,
                                 toolchain::StringRef dir_path = "");

// Prints the pass pipeline of `pass_manager` to `os`.
void PrintPassPipeline(const mlir::PassManager& pass_manager,
                       mlir::Operation* op, toolchain::raw_ostream& os);
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_MLIR_UTIL_H_
