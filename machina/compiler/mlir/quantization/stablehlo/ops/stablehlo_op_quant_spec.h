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

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_

#include <memory>

#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

// Returns StableHLO quantization specs for an op.
std::unique_ptr<OpQuantSpec> GetStableHloOpQuantSpec(Operation* op);

// Returns quantization constraints (ex: fixed output, same scale) given
// a StableHLO op.
std::unique_ptr<OpQuantScaleSpec> GetStableHloQuantConstraints(Operation* op);

// Checks if an op is quantizable in StableHLO quantizer. Argument op is not
// necessarily a StableHLO op.
bool IsOpQuantizableStableHlo(Operation* op);

}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_
