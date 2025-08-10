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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TF_UNIFORM_QUANTIZED_TYPES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TF_UNIFORM_QUANTIZED_TYPES_H_

#include <cstdint>

#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace tf_quant {

// Creates a `UniformQuantizedType` with the given `scale` and `zero_point`
// values. The produced type has f32 as its expressed type and i8 as its
// storage type. The available values use the full range of the storage value,
// i.e. [-128, 127]. Assumes asymmetric quantization, meaning the zero point
// value can be a non-zero value.
// If `narrow_range` is set true (ex: for weights), a restricted range of
// integers will be used for symmetric mapping, i.e. [-127, 127].
quant::UniformQuantizedType CreateI8F32UniformQuantizedType(
    Location loc, MLIRContext& context, double scale, int64_t zero_point,
    bool narrow_range = false);

// Creates a `UniformQuantizedType` with the given `scale` and `zero_point`
// values. The produced type has f32 as its expressed type and i32 as its
// storage type. The available values use the full range of the storage value.
// Assumes asymmetric quantization, meaning the zero point value can be
// a non-zero value.
quant::UniformQuantizedType CreateI32F32UniformQuantizedType(
    Location loc, MLIRContext& context, double scale, int64_t zero_point);

// Creates a `UniformQuantizedPerAxisType` with the given `scales` and
// `zero_points` values. The produced type has f32 as its expressed type and
// i8 as its storage type. The available values use the full range of the
// storage value, i.e. [-128, 127]. Assumes asymmetric quantization, meaning the
// zero point values can be non-zero values.
// If `narrow_range` is set true (ex: for weights), a restricted range of
// integers will be used for symmetric mapping, i.e. [-127, 127].
quant::UniformQuantizedPerAxisType CreateI8F32UniformQuantizedPerAxisType(
    Location loc, MLIRContext& context, ArrayRef<double> scales,
    ArrayRef<int64_t> zero_points, int quantization_dimension,
    bool narrow_range = false);

// Creates a `UniformQuantizedPerAxisType` with the given `scales` and
// `zero_points` values. The produced type has f32 as its expressed type and
// i32 as its storage type. The available values use the full range of the
// storage value. Assumes asymmetric quantization, meaning the
// zero point values can be non-zero values.
quant::UniformQuantizedPerAxisType CreateI32F32UniformQuantizedPerAxisType(
    Location loc, MLIRContext& context, ArrayRef<double> scales,
    ArrayRef<int64_t> zero_points, int quantization_dimension);

bool IsStorageTypeI8(quant::QuantizedType quantized_type);

bool IsStorageTypeI32(quant::QuantizedType quantized_type);

bool IsExpressedTypeF32(quant::QuantizedType quantized_type);

// Given a value, extract the `ElementType`.
// `value` should be a non-null `TensorType`.
inline Type GetElementType(const Value value) {
  return mlir::cast<TensorType>(value.getType()).getElementType();
}

// Returns true iff `type` is a uniform quantized type whose storage type is
// 8-bit integer and expressed type is f32.
bool IsI8F32UniformQuantizedType(Type type);

// Returns true iff `type` is a uniform quantized per-axis (per-channel) type
// whose storage type is 8-bit integer and expressed type is f32.
bool IsI8F32UniformQuantizedPerAxisType(Type type);

// Returns true iff `type` is a uniform quantized type whose storage type is
// 32-bit integer and expressed type is f32.
bool IsI32F32UniformQuantizedType(Type type);

// Returns true iff `type` is a uniform quantized per-axis (per-channel) type
// whose storage type is 32-bit integer and expressed type is f32.
bool IsI32F32UniformQuantizedPerAxisType(Type type);

// Determines whether the storage type of a quantized type is supported by
// `tfl.quantize` or `tfl.dequantize` ops. ui8, i8 and i16 are supported.
bool IsSupportedByTfliteQuantizeOrDequantizeOps(IntegerType storage_type);

// Returns true if a type is quantized tensor type.
bool IsQuantizedTensorType(Type type);

// Returns true if all operands and results are quantized.
bool IsOpFullyQuantized(Operation* op);

// Returns true iff none among operand and result tensors are quantized.
bool IsOpNotQuantized(Operation* op);

}  // namespace tf_quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TF_UNIFORM_QUANTIZED_TYPES_H_
