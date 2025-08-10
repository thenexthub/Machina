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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_REWRITE_UTIL
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_REWRITE_UTIL

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class GetScalarOfType<int value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class IsConstantValueOf<int value> : Constraint<CPred<
  "IsConstantValueOf($0," # value # ")">>;

def IsOnGpuDevicePred : CPred<"IsOnGpuDevice($0.getOwner())">;

#endif // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_REWRITE_UTIL
