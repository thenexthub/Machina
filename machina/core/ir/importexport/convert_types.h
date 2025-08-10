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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/statusor.h"

namespace mlir {
namespace tfg {
// Converts the TensorFlow DataType 'dtype' into an MLIR (scalar) type.
absl::Status ConvertDataType(machina::DataType dtype, Builder& builder,
                             Type* type);

// Converts a scalar MLIR type to a TensorFlow Datatype.
absl::Status ConvertScalarTypeToDataType(Type type,
                                         machina::DataType* dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
absl::Status ConvertToDataType(Type type, machina::DataType* dtype);

// Converts an TensorFlow shape to the one used in MLIR.
void ConvertToMlirShape(const machina::TensorShape& input_shape,
                        SmallVectorImpl<int64_t>* shape);

// Converts an TensorFlow shape proto to the one used in MLIR.
absl::Status ConvertToMlirShape(const machina::TensorShapeProto& input_shape,
                                SmallVectorImpl<int64_t>* shape);

// Given a tensor shape and dtype, get the corresponding MLIR tensor type.
absl::StatusOr<Type> ConvertToMlirTensorType(
    const machina::TensorShapeProto& shape, machina::DataType dtype,
    Builder* builder);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
