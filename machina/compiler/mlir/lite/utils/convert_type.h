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

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/schema/schema_generated.h"
#include "machina/core/framework/types.pb.h"

namespace mlir {
class Builder;
}  // namespace mlir

namespace tflite {
// Convert the MLIR type to the corresponding TFLite tensor.
tflite::TensorType ConvertTypeToTensorType(mlir::Type type);

// Convert the scalar type of a TFlite tensor to the corresponding MLIR type.
mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder);

// Convert the scalar type of a TFLite tensor to the corresponding
// Tensorflow type
machina::DataType TflTypeToTfType(tflite::TensorType type);

// Convert the Tensorflow scalar type to the corresponding TFLite type
absl::StatusOr<tflite::TensorType> TfTypeToTflType(machina::DataType type);

// Returns element type from attribute Type 'type_attr'.
mlir::Type GetShapeStrippedType(mlir::TypeAttr type_attr);

// Returns true if 'val' is not from Quantize op or
// from Quantize Op with same quant type as 'qtype_attr'
bool NotFromQuantOpOrSameQuantType(mlir::Value val, mlir::TypeAttr qtype_attr);

}  // namespace tflite
#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_
