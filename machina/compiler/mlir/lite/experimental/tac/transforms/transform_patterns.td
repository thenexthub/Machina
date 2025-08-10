/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "machina/compiler/mlir/lite/ir/tfl_ops.td"
include "machina/compiler/mlir/machina/ir/tf_ops.td"

// Basically x - y => x + y * -1
def SubToAdd : Pat<(TFL_SubOp $lhs, $rhs, $act),
  (TFL_AddOp $lhs, (TFL_MulOp $rhs,
                  (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>,
                   "-1.0f">), TFL_AF_None), $act)>;

// Squash tfl.dequantize and tfl.quantize pairs.
// TODO(b/185915462): Compare the scale of input and output. This can also be
// squashed to a requantize op if the scales are different.
def FoldQuantizeDequantize: Pat<(TFL_DequantizeOp (TFL_QuantizeOp $arg, $qt)), (replaceWithValue $arg)>;
