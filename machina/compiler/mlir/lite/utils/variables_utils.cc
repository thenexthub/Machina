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
#include "machina/compiler/mlir/lite/utils/variables_utils.h"

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace mlir {
namespace TFL {
namespace utils {

bool IsSupportedVariableType(Operation* op) {
  ShapedType type;
  if (toolchain::isa<TF::ReadVariableOp>(op)) {
    type = toolchain::cast<ShapedType>(op->getResult(0).getType());
  } else if (toolchain::isa<TF::AssignVariableOp>(op)) {
    type = toolchain::cast<ShapedType>(op->getOperand(1).getType());
  } else if (toolchain::isa<TF::VarHandleOp>(op)) {
    type =
        toolchain::cast<tf_type::TensorFlowTypeWithSubtype>(
            toolchain::cast<ShapedType>(op->getResult(0).getType()).getElementType())
            .GetSubtypes()
            .back();
  }
  return IsSupportedVariableType(type);
}

bool IsSupportedVariableType(ShapedType type) {
  auto element_type = type.getElementType();
  // Check complex types.
  if (auto complex_type = toolchain::dyn_cast<ComplexType>(element_type)) {
    auto complex_element_type = complex_type.getElementType();
    if (complex_element_type.isF32() || complex_element_type.isF64())
      return true;
  }
  // Check quantized types.
  if (auto quant_type = toolchain::dyn_cast<quant::QuantizedType>(element_type)) {
    // TFLite supports QI16, QI32, QI8, and QUI8
    if ((quant_type.getStorageTypeIntegralWidth() == 16 &&
         quant_type.isSigned()) ||
        quant_type.getStorageTypeIntegralWidth() == 8 ||
        (quant_type.getStorageTypeIntegralWidth() == 32 &&
         quant_type.isSigned()))
      return true;
  }
  return element_type.isF32() || element_type.isF64() ||
         element_type.isInteger(1) || element_type.isInteger(8) ||
         element_type.isInteger(32) || element_type.isSignlessInteger(64);
}

}  // namespace utils
}  // namespace TFL
}  // namespace mlir
