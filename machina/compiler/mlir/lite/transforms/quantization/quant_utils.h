/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::TFL {

inline constexpr char kPropagatedQuantizeOpAttr[] = "propagated";

std::optional<quant::QuantizedType> GetPropagatedType(
    SameScalesOpInterface same_scales_op);

// If `value` is the result of a DequantizeOp, returns the quantized type of the
// DequantizeOp's input. Otherwise, returns std::nullopt.
// The IR pattern looks like:
// ... -> [quantized type] -> DequantizeOp -> [value]
// Otherwise, returns std::nullopt.
std::optional<quant::QuantizedType> GetQTypeFromDefiningDequantize(
    mlir::Value value);

// If `value` has only one use and that use is a QuantizeOp, returns the
// quantized type of the QuantizeOp's result. Otherwise, returns std::nullopt.
// The single-use check is to avoid ambiguity in cases of fan-out.
// The IR pattern looks like:
// [value] -> QuantizeOp -> ...
std::optional<quant::QuantizedType> GetQTypeFromConsumingQuantize(
    mlir::Value value);

// Inserts a Quantize-Dequantize (QDQ) pair for a value.
// If `target_op` is provided, it only replaces the uses of `value` within
// `target_op`. Otherwise, it replaces all uses of `value` (except for the
// newly created Quantize op).
LogicalResult InsertQDQ(mlir::Value value, quant::QuantizedType qtype,
                        PatternRewriter& rewriter,
                        mlir::Operation* target_op = nullptr);

}  // namespace mlir::TFL

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_
