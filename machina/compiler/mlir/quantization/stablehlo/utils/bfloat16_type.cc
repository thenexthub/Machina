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

#include "machina/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir::quant::stablehlo {

bool IsLargeFloatType(Type type) {
  type = getElementTypeOrSelf(type);
  return isa<FloatType>(type) && type.getIntOrFloatBitWidth() > 16;
}

Type ToBfloat16Type(Type type) {
  if (auto shaped = mlir::dyn_cast<ShapedType>(type)) {
    const Type elem = shaped.getElementType();
    if (IsLargeFloatType(elem)) {
      return shaped.clone(BFloat16Type::get(type.getContext()));
    }
  } else if (IsLargeFloatType(type)) {
    return BFloat16Type::get(type.getContext());
  }
  return type;
}

}  // namespace mlir::quant::stablehlo
