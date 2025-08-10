/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

// Creates a pass that quantizes weight component of StableHLO graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizeWeightPass(
    const ::stablehlo::quantization::QuantizationComponentSpec&
        quantization_component_spec = {});

// Converts a serialized StableHLO module to bfloat16 and output serialized
// module.
absl::StatusOr<std::string> ConvertSerializedStableHloModuleToBfloat16(
    StringRef serialized_stablehlo_module);

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsPass(
    const ::stablehlo::quantization::QuantizationSpecs& quantization_specs);

// Creates a pass that inserts CalibrationStatisticsSaverOp.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCalibrationStatisticsSaverPass(
    StringRef calibration_data_dir,
    const std::vector<std::string>& aggregator_ops_to_ignore);

// Adds generated pass default constructors or options definitions.
#define GEN_PASS_DECL
// Adds generated pass registration functions.
#define GEN_PASS_REGISTRATION
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_
