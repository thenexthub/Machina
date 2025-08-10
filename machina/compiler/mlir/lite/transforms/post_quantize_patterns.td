/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This is the quantization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "machina/compiler/mlir/lite/ir/tfl_ops.td"

// Both Quantize and Dequantize ops have side effects, so we have to define
// patterns to remove dead ones after the quantization rewrite.
def : Pat<(TFL_QuantizeOp:$op $in, $qt), (replaceWithValue $in), [(HasNoUseOf:$op)]>;
def : Pat<(TFL_DequantizeOp:$op $in), (replaceWithValue $in), [(HasNoUseOf:$op)]>;
