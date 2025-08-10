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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_INSTRUMENTATIONS_SAVE_REPORT_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_INSTRUMENTATIONS_SAVE_REPORT_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassInstrumentation.h"  // part of Codira Toolchain

namespace mlir::quant::stablehlo {

// A `PassInstrumentation` that saves quantization report to file after
// `QuantizeCompositeFunctionsPass` is run. It inspects the `ModuleOp` after
// quantization and analyzes the quantizable units and quantization methods
// used. The report file will be saved at the `file_path`. The report file
// contains textproto of `QuantizationResults`. `file_path`'s base directories
// should exist (this pass instrumentation will not `mkdir` them).
//
// See `QuantizationReport` for further details on the quantization report.
class SaveQuantizationReportInstrumentation : public PassInstrumentation {
 public:
  // `file_path` is the path to save the report file. The report file is in
  // textproto format so a `.txtpb` extension is preferred but it doesn't result
  // in error if other extension is used. This instrumentation will not be run
  // if `file_path` is a `nullopt`.
  explicit SaveQuantizationReportInstrumentation(
      std::optional<absl::string_view> file_path);

  void runAfterPass(Pass* pass, Operation* op) override;

 private:
  std::optional<std::string> file_path_;  // Path to file to save the report.
};

}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_INSTRUMENTATIONS_SAVE_REPORT_H_
