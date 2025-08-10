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
#include "machina/compiler/mlir/quantization/common/uniform_quantized_types.h"

#include <cstdint>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Debug.h"
#include "toolchain/Support/MathExtras.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

#define DEBUG_TYPE "uniform-quantized-types"

namespace mlir {
namespace quant {

UniformQuantizedType CreateI8F32UniformQuantizedType(const Location loc,
                                                     MLIRContext& context,
                                                     const double scale,
                                                     const int64_t zero_point,
                                                     const bool narrow_range) {
  return UniformQuantizedType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/8),
      /*expressedType=*/Float32Type::get(&context), scale, zero_point,
      /*storageTypeMin=*/toolchain::minIntN(8) + (narrow_range ? 1 : 0),
      /*storageTypeMax=*/toolchain::maxIntN(8));
}

UniformQuantizedType CreateI32F32UniformQuantizedType(
    const Location loc, MLIRContext& context, const double scale,
    const int64_t zero_point) {
  return UniformQuantizedType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/32),
      /*expressedType=*/Float32Type::get(&context), scale, zero_point,
      /*storageTypeMin=*/toolchain::minIntN(32),
      /*storageTypeMax=*/toolchain::maxIntN(32));
}

UniformQuantizedPerAxisType CreateI8F32UniformQuantizedPerAxisType(
    const Location loc, MLIRContext& context, const ArrayRef<double> scales,
    const ArrayRef<int64_t> zero_points, const int quantization_dimension,
    const bool narrow_range) {
  return UniformQuantizedPerAxisType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/8),
      /*expressedType=*/Float32Type::get(&context),
      SmallVector<double>(scales), SmallVector<int64_t>(zero_points),
      quantization_dimension,
      /*storageTypeMin=*/toolchain::minIntN(8) + (narrow_range ? 1 : 0),
      /*storageTypeMax=*/toolchain::maxIntN(8));
}

UniformQuantizedPerAxisType CreateI32F32UniformQuantizedPerAxisType(
    const Location loc, MLIRContext& context, const ArrayRef<double> scales,
    const ArrayRef<int64_t> zero_points, const int quantization_dimension) {
  return UniformQuantizedPerAxisType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/32),
      /*expressedType=*/Float32Type::get(&context),
      SmallVector<double>(scales), SmallVector<int64_t>(zero_points),
      quantization_dimension, /*storageTypeMin=*/toolchain::minIntN(32),
      /*storageTypeMax=*/toolchain::maxIntN(32));
}

bool IsStorageTypeI8(const QuantizedType quantized_type) {
  const Type storage_type = quantized_type.getStorageType();
  return storage_type.isInteger(/*width=*/8);
}

bool IsStorageTypeI32(const QuantizedType quantized_type) {
  const Type storage_type = quantized_type.getStorageType();
  return storage_type.isInteger(/*width=*/32);
}

bool IsExpressedTypeF32(const QuantizedType quantized_type) {
  const Type expressed_type = quantized_type.getExpressedType();
  return mlir::isa<Float32Type>(expressed_type);
}

bool IsI8F32UniformQuantizedType(const Type type) {
  const UniformQuantizedType quantized_type =
      mlir::dyn_cast_or_null<UniformQuantizedType>(type);
  if (!quantized_type) {
    LLVM_DEBUG(toolchain::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI8(quantized_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an i8 storage type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  return true;
}

bool IsI8F32UniformQuantizedPerAxisType(const Type type) {
  const UniformQuantizedPerAxisType quantized_per_axis_type =
      mlir::dyn_cast_or_null<UniformQuantizedPerAxisType>(type);
  if (!quantized_per_axis_type) {
    LLVM_DEBUG(toolchain::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI8(quantized_per_axis_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an i8 storage type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_per_axis_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  return true;
}

bool IsI32F32UniformQuantizedType(const Type type) {
  const UniformQuantizedType quantized_type =
      mlir::dyn_cast_or_null<UniformQuantizedType>(type);
  if (!quantized_type) {
    LLVM_DEBUG(toolchain::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI32(quantized_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an i32 storage type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  return true;
}

bool IsI32F32UniformQuantizedPerAxisType(const Type type) {
  const UniformQuantizedPerAxisType quantized_per_axis_type =
      mlir::dyn_cast_or_null<UniformQuantizedPerAxisType>(type);
  if (!quantized_per_axis_type) {
    LLVM_DEBUG(toolchain::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI32(quantized_per_axis_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an i32 storage type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_per_axis_type)) {
    LLVM_DEBUG(toolchain::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  return true;
}

// Determines whether the storage type of a quantized type is supported by
// `tfl.quantize` or `tfl.dequantize` ops. ui8, i8 and i16 are supported.
bool IsSupportedByTfliteQuantizeOrDequantizeOps(IntegerType storage_type) {
  if (storage_type.getWidth() == 8 ||
      (storage_type.isSigned() && storage_type.getWidth() == 16)) {
    return true;
  }
  LLVM_DEBUG(toolchain::dbgs()
             << "Uniform quantize / dequantize op only supports ui8, i8 or "
                "i16 for the storage type of uniform quantized type. Got: "
             << storage_type << ".\n");
  return false;
}

bool IsQuantizedTensorType(Type type) {
  if (!mlir::isa<TensorType>(type)) {
    return false;
  }
  Type element_type = mlir::cast<TensorType>(type).getElementType();
  return mlir::isa<QuantizedType>(element_type);
}

bool IsOpFullyQuantized(Operation* op) {
  return toolchain::all_of(op->getOperandTypes(), IsQuantizedTensorType) &&
         toolchain::all_of(op->getResultTypes(), IsQuantizedTensorType);
}

bool IsOpNotQuantized(Operation* op) {
  return !toolchain::any_of(op->getOperandTypes(), IsQuantizedTensorType) &&
         !toolchain::any_of(op->getResultTypes(), IsQuantizedTensorType);
}

}  // namespace quant
}  // namespace mlir
