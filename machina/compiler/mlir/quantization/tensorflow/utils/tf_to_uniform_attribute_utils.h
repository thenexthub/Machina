/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
// This header file defines common utils used when transforming TF ops to
// Uniform Quantized ops.

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_

#include "toolchain/ADT/StringMap.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"

namespace mlir::quant {

LogicalResult FillAttributesForUniformQuantizedDotOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedConvolutionOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedAddOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedClipByValueOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformRequantizeOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizeOp(
    PatternRewriter& rewriter, Operation* op,
    toolchain::StringMap<Attribute>& identifier_to_attr,
    machina::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

}  // namespace mlir::quant

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_
