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

#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_

#include "toolchain/Support/Casting.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CastInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace tac {

// Returns true if 'op' is non const op. Returns false otherwise or if
// 'op' is null.
inline bool IsNonConstOp(Operation* op) {
  if (!op) return false;
  if (toolchain::isa<arith::ConstantOp, mlir::func::ConstantOp>(op)) return false;
  if (op->hasTrait<OpTrait::ConstantLike>()) return false;
  if (toolchain::isa<TFL::ConstOp, TFL::QConstOp>(op)) return false;
  return true;
}

// Returns true if 'op' is a terminator op, otherwise false.
bool IsTerminatorOp(Operation* op);

// Returns true if 'op' is not TFL Quant / Dequant op. Returns False otherwise
// or if 'op' is null.
bool NotTFLQuantDequantizeOp(Operation* op);

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t) {
  if (auto shaped_type = mlir::dyn_cast_or_null<ShapedType>(t)) {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Return true when the given element_type is QI8.
inline bool IsQI8Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         quantized_type.isSigned();
}

// Return true when the given element_type is QUI8.
inline bool IsQUI8Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         !quantized_type.isSigned();
}

// Return true when the given element_type is QI32.
inline bool IsQI32Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 32 &&
         quantized_type.isSigned();
}

// Try to guess the inference type of the op.
InferenceType GetInferenceType(Operation* op);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_
