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

// This file provides a list of supported quantization algorithms in the format
// of "apply<Name of the Quantization Algorithm>Quantization".
// After applying the function, a quantize/dequantize functions are created
// where the body of each function contains a specific quantization algorithm.
// The input of the quantize function has one operand of
// IsValueWithQuantizablePrecision and the output is a tensor with supported
// quantized precision (like int8). For dequantize function, it is the other way
// around.

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_OPS_TF_QUANTIZE_OP_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_OPS_TF_QUANTIZE_OP_H_

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Traits.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {

std::optional<TF::PartitionedCallOp> ApplyUniformQuantization(
    PatternRewriter& rewriter, TF::ConstOp op,
    machina::quantization::QuantizationComponentSpec& weight_spec);

}  // namespace quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_OPS_TF_QUANTIZE_OP_H_
