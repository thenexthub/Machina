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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_TENSOR_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_TENSOR_H_

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/protobuf/struct.pb.h"

namespace machina {

using tsl::StatusOr;

// Converts an TensorFlow tensor proto into an MLIR elements attribute.
absl::StatusOr<mlir::ElementsAttr> ConvertTensorProto(
    const TensorProto& input_tensor, mlir::Builder* builder,
    bool convert_to_dense_resource = false);

// Converts an TensorFlow tensor into an MLIR elements attribute.
absl::StatusOr<mlir::ElementsAttr> ConvertTensor(
    const Tensor& input_tensor, mlir::Builder* builder,
    bool convert_to_dense_resource = false);

// Converts a shape from MLIR to a TensorFlow tensor shape proto.
void ConvertToTensorShapeProto(toolchain::ArrayRef<int64_t> shape,
                               TensorShapeProto* output_shape);

// Converts an MLIR type to a TensorFlow tensor shape.
PartialTensorShape ConvertTypeToTensorShape(const mlir::Type& type);

// Converts an MLIR shaped type to a TensorFlow shape attribute.
mlir::TF::ShapeAttr ConvertTypeToTensorShapeAttr(const mlir::Type& type);

// Converts an MLIR shaped type to a Tensorflow tensor spec proto.
absl::StatusOr<TensorSpecProto> ConvertTypeToTensorSpecProto(
    const mlir::Type& type);

// Converts a TensorFlow shape attribute to an MLIR shape attribute.
absl::StatusOr<mlir::Attribute> ConvertTensorShapeProto(
    const TensorShapeProto& shape, mlir::MLIRContext* context);

// Converts an MLIR elements attribute to a TensorFlow tensor proto.
absl::Status ConvertToTensorProto(mlir::ElementsAttr attr,
                                  TensorProto* output_tensor);

// Converts an MLIR elements attribute to a TensorFlow tensor.
absl::Status ConvertToTensor(mlir::ElementsAttr attr, Tensor* output_tensor);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_TENSOR_H_
