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

// List of quantization DRRs that are to be used when we want the converter to
// take authoritative decisions about quantization as opposed to strictly
// following Q-DQ annotations.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/IR/CommonTypeConstraints.td"
include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization.td"
include "machina/compiler/mlir/lite/ir/tfl_ops.td"

// Transpose conv supports hybrid computation with quantized weights.
def FoldQuantWeightsIntoTposeConv : Pat<
  (TFL_TransposeConvOp
    $output_shape,
    (TFL_DequantizeOp $quant_weights),
    $quant_input,
    $bias, $padding, $stride_h, $stride_w, $faf),
  (TFL_TransposeConvOp $output_shape, $quant_weights,
    $quant_input, $bias, $padding, $stride_h, $stride_w, $faf), 
  [(TensorOf<[QI8, QUI8, QI16]> $quant_weights)]>;

// EmbeddingLookup supports quantized table tensors.
def FoldQuantWeightsIntoEmbeddingLookup : Pat<
  (TFL_EmbeddingLookupOp:$output $indices, (TFL_DequantizeOp $table)),
  (TFL_EmbeddingLookupOp $indices, $table),
  [(TensorOf<[QI4, QI8, QUI8]> $table), (TensorOf<[F32]> $output)]>;
