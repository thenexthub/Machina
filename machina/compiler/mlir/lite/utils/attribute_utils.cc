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

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {

FloatAttr ExtractSingleElementAsFloat(ElementsAttr attr) {
  if (attr.getShapedType().getNumElements() != 1 ||
      !mlir::isa<FloatType>(attr.getShapedType().getElementType())) {
    return {};
  }
  return attr.getSplatValue<FloatAttr>();
}

FloatAttr GetSingleElementAsFloatOrSelf(Attribute attr) {
  if (auto m = mlir::dyn_cast_or_null<ElementsAttr>(attr)) {
    return ExtractSingleElementAsFloat(m);
  } else {
    return mlir::dyn_cast_or_null<FloatAttr>(attr);
  }
}

IntegerAttr ExtractSingleElementAsInteger(ElementsAttr attr) {
  if (attr.getShapedType().getNumElements() != 1 ||
      !attr.getShapedType().getElementType().isSignlessInteger()) {
    return {};
  }
  return attr.getSplatValue<IntegerAttr>();
}

}  // namespace TFL
}  // namespace mlir
