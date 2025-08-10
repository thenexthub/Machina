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
#include "machina/compiler/mlir/machina/utils/dynamic_shape_utils.h"

#include "toolchain/ADT/SmallVector.h"

namespace machina {

toolchain::SmallVector<int64_t> ConvertTFShapeToMlir(
    toolchain::ArrayRef<int64_t> shapes) {
  return toolchain::to_vector(toolchain::map_range(shapes, [](int64_t shape) {
    return shape == kTFDynamicSize ? mlir::ShapedType::kDynamic : shape;
  }));
}

toolchain::SmallVector<int64_t> ConvertMlirShapeToTF(
    toolchain::ArrayRef<int64_t> shapes) {
  return toolchain::to_vector(toolchain::map_range(shapes, [](int64_t shape) {
    return mlir::ShapedType::isDynamic(shape) ? kTFDynamicSize : shape;
  }));
}

mlir::RankedTensorType GetTypeFromTFTensorShape(toolchain::ArrayRef<int64_t> shape,
                                                mlir::Type elementType,
                                                mlir::Attribute encoding) {
  return mlir::RankedTensorType::get(ConvertTFShapeToMlir(shape), elementType,
                                     encoding);
}

}  // namespace machina
