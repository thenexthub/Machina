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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/types/dialect.h"
#include "machina/core/platform/statusor.h"

namespace mlir {
namespace tfg {

// Converts an TensorFlow tensor proto into an MLIR elements attribute.
absl::StatusOr<ElementsAttr> ConvertTensorProto(
    const machina::TensorProto& input_tensor, Builder builder);

// Converts an TensorFlow tensor into an MLIR elements attribute.
absl::StatusOr<ElementsAttr> ConvertTensor(
    const machina::Tensor& input_tensor, Builder builder);

// Converts a shape from MLIR to a TensorFlow tensor shape proto.
void ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                               machina::TensorShapeProto* output_shape);

// Converts an MLIR type to a TensorFlow tensor shape.
machina::PartialTensorShape ConvertTypeToTensorShape(const Type& type);

// Converts a TensorFlow shape attribute to an MLIR shape attribute.
absl::StatusOr<ShapeAttr> ConvertTensorShapeProto(
    const machina::TensorShapeProto& shape, MLIRContext* context);

// Fill in the contents of TensorShapeProto for the given shape.
// ShapeContainerT is any type with the following methods:
//   bool hasRank()
//   ArrayRef<int64_t> getShape()
// This includes TF::ShapeAttr and ShapedType.
template <typename ShapeContainerT>
void SetTensorShapeProto(ShapeContainerT shape,
                         machina::TensorShapeProto* proto) {
  if (shape.hasRank()) {
    for (int64_t dim : shape.getShape()) {
      // TODO(hinsu): Use machina::kTFDynamicSize instead of -1 without
      // depending on machina/compiler
      proto->add_dim()->set_size(mlir::ShapedType::isDynamic(dim) ? -1 : dim);
    }
  } else {
    proto->set_unknown_rank(true);
  }
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto.
absl::Status ConvertToTensorProto(ElementsAttr attr,
                                  machina::TensorProto* output_tensor);

// Converts an MLIR elements attribute to a TensorFlow tensor.
absl::Status ConvertToTensor(ElementsAttr attr,
                             machina::Tensor* output_tensor);

// Converts a TF shape to MLIR shape, i.e. -1 becomes kDynamicSize.
toolchain::SmallVector<int64_t> ConvertTFShapeToMlir(toolchain::ArrayRef<int64_t> shape);

// Converts an MLIR shape to TF shape, i.e. kDynamicSize becomes -1.
toolchain::SmallVector<int64_t> ConvertMlirShapeToTF(toolchain::ArrayRef<int64_t> shape);

// Creates a TF TensorShape using MLIR shape, element type and encoding.
mlir::RankedTensorType GetTypeFromTFTensorShape(toolchain::ArrayRef<int64_t> shape,
                                                mlir::Type elementType,
                                                mlir::Attribute encoding = {});

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_
