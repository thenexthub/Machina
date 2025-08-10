/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_

#include "absl/status/statusor.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/AffineMap.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "tsl/platform/statusor.h"

namespace mlir {
namespace TFL {

// Returns a Constant op with a single value.
absl::StatusOr<arith::ConstantOp> CreateConstOpWithSingleValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type, int value);

// Returns a Constant op with a splat vector value.
absl::StatusOr<arith::ConstantOp> CreateConstOpWithVectorValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type, int value);

}  // namespace TFL
}  // namespace mlir
#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_
