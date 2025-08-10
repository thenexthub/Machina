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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "machina/compiler/mlir/quantization/common/attrs_and_constraints.td"
include "machina/compiler/mlir/machina/ir/tf_ops.td"

// Remove redundant `CastOp` to int8 if the input is properly clipped.
def RemoveRedundantCastOps : Pat<
  (TF_CastOp:$root_cast
    (TF_CastOp:$i8_cast
      (TF_ClipByValueOp:$clip $input, $min_value, $max_value),
      ConstBoolAttrFalse:$truncate2),
    ConstBoolAttrFalse:$truncate1),
  (TF_CastOp $clip, ConstBoolAttrFalse),
  [(TensorOf<[I8]> $i8_cast),
   (TensorOf<[I32]> $clip),
   (IsIntSplatValueEqual<"int32_t", "-128"> $min_value),
   (IsIntSplatValueEqual<"int32_t", "127"> $max_value)]>;

// This pattern optimizes:
//   (x + cst1) + cst2 -> x + cst
//   (x - cst1) - cst2 -> x - cst
// Where: cst = cst1 + cst2
foreach BinaryOp = [TF_AddV2Op, TF_SubOp] in {
  def OptimizeConsecutive#BinaryOp : Pat<
    (BinaryOp
      (BinaryOp $x, (TF_ConstOp:$cst1 $cst1_value)),
      (TF_ConstOp:$cst2 $cst2_value)),
    (BinaryOp
      $x, (TF_AddV2Op $cst1, $cst2))>;
}

// This pattern optimizes:
//   (x + cst1) - cst2 -> x - cst
//   (x - cst1) + cst2 -> x + cst
// Where: cst = cst2 - cst1
foreach BinaryOpPair = [[TF_AddV2Op, TF_SubOp],
                     [TF_SubOp, TF_AddV2Op]] in {
  def OptimizeConsecutive#BinaryOpPair[0]#BinaryOpPair[1] : Pat<
    (BinaryOpPair[0]
      (BinaryOpPair[1] $x, (TF_ConstOp:$cst1 $cst1_value)),
      (TF_ConstOp:$cst2 $cst2_value)),
    (BinaryOpPair[0]
      $x, (TF_SubOp  $cst2, $cst1))>;
}
