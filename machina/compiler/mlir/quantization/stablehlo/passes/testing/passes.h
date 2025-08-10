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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_TESTING_PASSES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_TESTING_PASSES_H_

#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

// Identifies predefined `QuantizationSpecs` for
// `TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass`. The pass
// option argument is specified in line comments for each enum value.
enum class TestQuantizationSpecs {
  kEmpty,                         // empty
  kDisableAllDotGeneral,          // disable-all-dot-general
  kStaticRangePtqToAll,           // static-range-ptq-to-all
  kStaticRangePtqToComputeHeavy,  // static-range-ptq-to-compute-heavy
};

// Adds generated pass default constructors or options definitions.
#define GEN_PASS_DECL
// Adds generated pass registration functions.
#define GEN_PASS_REGISTRATION
#include "machina/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

}  // namespace mlir::quant::stablehlo::testing

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_TESTING_PASSES_H_
