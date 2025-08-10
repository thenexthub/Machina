/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace mlir::quant::machina {

// GetDenseAttrFromTensorProtoAttr returns DenseElementsAttr from tensor proto.
FailureOr<mlir::DenseElementsAttr> GetDenseAttrFromTensorProtoAttr(
    toolchain::StringRef mangled_tensor_proto, TensorType result_tensor_type);

// Check if a type is TF qint type.
bool IsTFQintType(Type type);

// Convert qint type to the corresponding int type. Return original type if it
// is not qint type.
Type GetIntTypeFromTFQint(Type type);

// Check if an op is TF UniformQuantized op.
bool IsTFUniformQuantizedOp(Operation* op);

}  // namespace mlir::quant::machina

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_
