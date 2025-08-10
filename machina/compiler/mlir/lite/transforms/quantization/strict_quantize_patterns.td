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

// Provides patterns for quantization following the framework annotations.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/IR/CommonTypeConstraints.td"
include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization.td"
include "machina/compiler/mlir/lite/ir/tfl_ops.td"

def HasSameType : Constraint<CPred<[{$0.getType() == $1.getType()}]>>;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def IsNotFromSameScalesOp : Constraint<CPred<
    "!$0.getDefiningOp() || !toolchain::isa<mlir::TFL::SameScalesOpInterface>($0.getDefiningOp())">>;

def RemoveNoOpQ: Pat<(TFL_QuantizeOp:$out $in, $qt),
          (replaceWithValue $in),
          [(HasSameType $in, $out)]>;

// Change DQ->Q with different types to a requant.
def FuseDqQToRequant: Pat<(TFL_QuantizeOp:$out (TFL_DequantizeOp:$mid $in), $qt),
          (TFL_QuantizeOp $in, $qt),
          [(HasOneUse $mid)]>;

// Squash repeating quantize ops.
// Unless the preceding op needs is strict about its scale requirements in which 
// case, we keep the first quantize to be fused later and let the second 
// quantize be(come) a requant op.
def FuseQQToRequant: Pat<(TFL_QuantizeOp (TFL_QuantizeOp:$mid $in, $_), $qt2),
          (TFL_QuantizeOp $in, $qt2),
          [(HasOneUse $mid), (IsNotFromSameScalesOp $in)]>;
