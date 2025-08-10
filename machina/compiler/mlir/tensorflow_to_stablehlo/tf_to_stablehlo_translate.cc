/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina_to_stablehlo/tf_to_stablehlo.h"

namespace {

using toolchain::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_path(toolchain::cl::Positional,
                            toolchain::cl::desc("<input path>"), toolchain::cl::Required);

// NOLINTNEXTLINE
opt<std::string> output_filename("o", toolchain::cl::desc("<output path>"),
                                 toolchain::cl::Optional, toolchain::cl::init("-"));

// NOLINTNEXTLINE
opt<std::string> input_arg_shapes_str(
    "input-arg-shapes",
    toolchain::cl::desc(
        "A string representation of input argument shapes for 'main' "
        "entry-point, separating tensors with ':', dimension with ',', and "
        "using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?' "
        "expresses argument shapes [1,2], [] and [1,?]"),
    toolchain::cl::Optional, toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> exported_model_signatures(
    "exported-model-signatures",
    toolchain::cl::desc(
        "Comma-separated list of exported model signatures to convert"),
    toolchain::cl::Optional, toolchain::cl::init("serving_default"));

// NOLINTNEXTLINE
opt<std::string> tag_names(
    "tags",
    toolchain::cl::desc("Comma-separated list of tags for loading SavedModel. "
                   "Ignored for MLIR input"),
    toolchain::cl::Optional, toolchain::cl::init("serve"));

// NOLINTNEXTLINE
opt<bool> elide_large_elements_attrs(
    "e",
    toolchain::cl::desc(
        "Elide large elements attrs while dumping the output StableHLO."),
    toolchain::cl::Optional, toolchain::cl::init(false));

}  // namespace

namespace mlir {

namespace {
// Dump the ModuleOp 'module' to the file specified using 'outputFileName'
absl::Status ExportModule(ModuleOp module) {
  std::string error_msg;
  auto output = openOutputFile(output_filename, &error_msg);
  if (output == nullptr) {
    return absl::AbortedError(
        absl::StrCat("Unable to write to output path: ", error_msg));
  }

  // Export StableHLO MLIR as output
  std::string result;
  toolchain::raw_string_ostream os(result);
  OpPrintingFlags printing_flags;
  if (elide_large_elements_attrs) {
    printing_flags.elideLargeElementsAttrs();
  }
  module.print(os, printing_flags);
  os.flush();

  output->os() << result;
  output->keep();

  return absl::OkStatus();
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  machina::InitMlir y(&argc, &argv);
  toolchain::cl::ParseCommandLineOptions(argc, argv,
                                    "TF Saved Model to Stablehlo converter\n");

  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool is_input_mlir_module = absl::EndsWith(input_path, ".mlir");
  std::vector<std::string> exported_model_signatures_in_vector =
      absl::StrSplit(exported_model_signatures, ',');
  std::vector<std::string> tag_names_in_vector = absl::StrSplit(tag_names, ',');
  auto module = TfToStablehlo(
      input_path, &context, exported_model_signatures_in_vector,
      tag_names_in_vector, input_arg_shapes_str, is_input_mlir_module);
  if (!module.ok()) {
    toolchain::errs() << module.status().ToString() << "\n";
    return module.status().raw_code();
  }

  return mlir::ExportModule(module->get()).raw_code();
}
