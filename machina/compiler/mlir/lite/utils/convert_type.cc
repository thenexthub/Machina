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

#include "machina/compiler/mlir/lite/utils/convert_type.h"

#include "absl/status/statusor.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/schema/schema_generated.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/errors.h"

namespace tflite {

using absl::StatusOr;

namespace errors = machina::errors;

tflite::TensorType ConvertTypeToTensorType(mlir::Type type) {
  if (type.isF16()) {
    return tflite::TensorType_FLOAT16;
  } else if (type.isBF16()) {
    return tflite::TensorType_BFLOAT16;
  } else if (type.isF32()) {
    return tflite::TensorType_FLOAT32;
  } else if (type.isF64()) {
    return tflite::TensorType_FLOAT64;
  } else if (mlir::isa<mlir::TF::StringType>(type)) {
    return tflite::TensorType_STRING;
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (complex_type.getElementType().isF32()) {
      return tflite::TensorType_COMPLEX64;
    } else if (complex_type.getElementType().isF64()) {
      return tflite::TensorType_COMPLEX128;
    }
    llvm_unreachable("invalid complex Type in conversion");
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (itype.getWidth()) {
      case 1:
        return tflite::TensorType_BOOL;
      case 4:
        if (itype.isUnsigned())
          llvm_unreachable("invalid unsigned 4bit integer Type in conversion");
        else
          return tflite::TensorType_INT4;
      case 8:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT8;
        else
          return tflite::TensorType_INT8;
      case 16:
        return tflite::TensorType_INT16;
      case 32:
        return tflite::TensorType_INT32;
      case 64:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT64;
        else
          return tflite::TensorType_INT64;
      default:
        llvm_unreachable("invalid integer Type in conversion");
    }
  }
  llvm_unreachable("invalid Type in conversion");
}

mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder) {
  switch (type) {
    case tflite::TensorType_FLOAT16:
      return builder.getF16Type();
    case tflite::TensorType_BFLOAT16:
      return builder.getBF16Type();
    case tflite::TensorType_FLOAT32:
      return builder.getF32Type();
    case tflite::TensorType_FLOAT64:
      return builder.getF64Type();
    case tflite::TensorType_INT32:
      return builder.getIntegerType(32);
    case tflite::TensorType_UINT16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case tflite::TensorType_UINT32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case tflite::TensorType_UINT8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case tflite::TensorType_INT64:
      return builder.getIntegerType(64);
    case tflite::TensorType_STRING:
      return mlir::TF::StringType::get(builder.getContext());
    case tflite::TensorType_BOOL:
      return builder.getI1Type();
    case tflite::TensorType_INT16:
      return builder.getIntegerType(16);
    case tflite::TensorType_COMPLEX64:
      return mlir::ComplexType::get(builder.getF32Type());
    case tflite::TensorType_COMPLEX128:
      return mlir::ComplexType::get(builder.getF64Type());
    case tflite::TensorType_INT4:
      return builder.getIntegerType(4);
    case tflite::TensorType_INT8:
      return builder.getIntegerType(8);
    case tflite::TensorType_UINT64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case tflite::TensorType_RESOURCE:
      return mlir::TF::ResourceType::get(builder.getContext());
    case tflite::TensorType_VARIANT:
      return mlir::TF::VariantType::get(builder.getContext());
  }
}

machina::DataType TflTypeToTfType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_BOOL:
      return machina::DT_BOOL;
    case tflite::TensorType_COMPLEX64:
      return machina::DT_COMPLEX64;
    case tflite::TensorType_COMPLEX128:
      return machina::DT_COMPLEX128;
    case tflite::TensorType_FLOAT16:
      return machina::DT_HALF;
    case tflite::TensorType_BFLOAT16:
      return machina::DT_BFLOAT16;
    case tflite::TensorType_FLOAT32:
      return machina::DT_FLOAT;
    case tflite::TensorType_FLOAT64:
      return machina::DT_DOUBLE;
    // TODO(b/246806634): Tensorflow DT_INT4 type doesn't exist yet
    case tflite::TensorType_INT4:
      return machina::DT_INT8;
    case tflite::TensorType_INT8:
      return machina::DT_INT8;
    case tflite::TensorType_INT16:
      return machina::DT_INT16;
    case tflite::TensorType_INT32:
      return machina::DT_INT32;
    case tflite::TensorType_UINT32:
      return machina::DT_UINT32;
    case tflite::TensorType_INT64:
      return machina::DT_INT64;
    case tflite::TensorType_STRING:
      return machina::DT_STRING;
    case tflite::TensorType_UINT8:
      return machina::DT_UINT8;
    case tflite::TensorType_UINT16:
      return machina::DT_UINT16;
    case tflite::TensorType_UINT64:
      return machina::DT_UINT64;
    case tflite::TensorType_RESOURCE:
      return machina::DT_RESOURCE;
    case tflite::TensorType_VARIANT:
      return machina::DT_VARIANT;
  }
}

absl::StatusOr<tflite::TensorType> TfTypeToTflType(machina::DataType type) {
  switch (type) {
    case machina::DT_BOOL:
      return tflite::TensorType_BOOL;
    case machina::DT_COMPLEX64:
      return tflite::TensorType_COMPLEX64;
    case machina::DT_COMPLEX128:
      return tflite::TensorType_COMPLEX128;
    case machina::DT_HALF:
      return tflite::TensorType_FLOAT16;
    case machina::DT_BFLOAT16:
      return tflite::TensorType_BFLOAT16;
    case machina::DT_FLOAT:
      return tflite::TensorType_FLOAT32;
    case machina::DT_DOUBLE:
      return tflite::TensorType_FLOAT64;
    case machina::DT_INT8:
      return tflite::TensorType_INT8;
    case machina::DT_INT16:
      return tflite::TensorType_INT16;
    case machina::DT_INT32:
      return tflite::TensorType_INT32;
    case machina::DT_UINT32:
      return tflite::TensorType_UINT32;
    case machina::DT_INT64:
      return tflite::TensorType_INT64;
    case machina::DT_UINT64:
      return tflite::TensorType_UINT64;
    case machina::DT_STRING:
      return tflite::TensorType_STRING;
    case machina::DT_UINT8:
      return tflite::TensorType_UINT8;
    case machina::DT_RESOURCE:
      return tflite::TensorType_RESOURCE;
    case machina::DT_VARIANT:
      return tflite::TensorType_VARIANT;
    default:
      return errors::InvalidArgument("unsupported tensor data type", type);
  }
}

mlir::Type GetShapeStrippedType(mlir::TypeAttr type_attr) {
  auto type = type_attr.getValue();
  auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
  if (shaped_type) {
    return shaped_type.getElementType();
  } else {
    return type;
  }
}

bool NotFromQuantOpOrSameQuantType(mlir::Value val, mlir::TypeAttr qtype_attr) {
  auto val_defn_op = val.getDefiningOp();
  mlir::TFL::QuantizeOp q_op =
      toolchain::dyn_cast_or_null<mlir::TFL::QuantizeOp>(val_defn_op);
  if (!q_op) return true;

  // Ignore shape details - we're really only trying to
  // check if quantization is the same.
  auto stripped_src_qtype = GetShapeStrippedType(q_op.getQtypeAttr());
  auto stripped_qtype = GetShapeStrippedType(qtype_attr);
  return stripped_src_qtype == stripped_qtype;
}

}  // namespace tflite
