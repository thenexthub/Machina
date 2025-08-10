/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/core/transforms/utils/op_cat_helper.h"

#include "absl/status/status.h"
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/ir/dialect.h"

namespace mlir {
namespace tfg {

namespace {

bool SplatElementsAttrHasValue(SplatElementsAttr attr, float v) {
  Type type = attr.getElementType();

#define IF_SPLAT_VALUE_IS(DTYPE, VALUE)                                \
  if (attr.getSplatValue<machina::EnumToDataType<DTYPE>::Type>() == \
      machina::EnumToDataType<DTYPE>::Type(VALUE))                  \
    return true;

  if (type.isInteger(1)) {
    IF_SPLAT_VALUE_IS(machina::DT_BOOL, v);
  } else if (type.isSignedInteger()) {
    if (type.isInteger(8)) {
      IF_SPLAT_VALUE_IS(machina::DT_INT8, v);
    } else if (type.isInteger(16)) {
      IF_SPLAT_VALUE_IS(machina::DT_INT16, v);
    } else if (type.isInteger(32)) {
      IF_SPLAT_VALUE_IS(machina::DT_INT32, v);
    } else if (type.isInteger(64)) {
      IF_SPLAT_VALUE_IS(machina::DT_INT64, v);
    }
  } else if (type.isUnsignedInteger()) {
    if (type.isInteger(8)) IF_SPLAT_VALUE_IS(machina::DT_UINT8, v);
    if (type.isInteger(16)) IF_SPLAT_VALUE_IS(machina::DT_UINT16, v);
  } else if (type.isF16()) {
    IF_SPLAT_VALUE_IS(machina::DT_HALF, v);
  } else if (type.isF32()) {
    IF_SPLAT_VALUE_IS(machina::DT_FLOAT, v);
  } else if (type.isF64()) {
    IF_SPLAT_VALUE_IS(machina::DT_DOUBLE, v);
  } else if (type.isBF16()) {
    IF_SPLAT_VALUE_IS(machina::DT_BFLOAT16, v);
  } else if (mlir::isa<ComplexType>(type)) {
    ComplexType complex_type = mlir::cast<ComplexType>(type);
    if (complex_type.getElementType().isF32()) {
      IF_SPLAT_VALUE_IS(machina::DT_COMPLEX64, v);
    } else if (complex_type.getElementType().isF64()) {
      IF_SPLAT_VALUE_IS(machina::DT_COMPLEX128, v);
    }
  } else if (mlir::isa<tf_type::Qint8Type>(type)) {
    IF_SPLAT_VALUE_IS(machina::DT_QINT8, v);
  } else if (mlir::isa<tf_type::Qint16Type>(type)) {
    IF_SPLAT_VALUE_IS(machina::DT_QINT16, v);
  } else if (mlir::isa<tf_type::Qint32Type>(type)) {
    IF_SPLAT_VALUE_IS(machina::DT_QINT32, v);
  } else if (mlir::isa<tf_type::Quint8Type>(type)) {
    IF_SPLAT_VALUE_IS(machina::DT_QUINT8, v);
  } else if (mlir::isa<tf_type::Quint16Type>(type)) {
    IF_SPLAT_VALUE_IS(machina::DT_QUINT16, v);
  }
#undef IF_SPLAT_VALUE_IS
  return false;
}

}  // namespace

bool OpCatHelper::IsAggregate(TFOp op) {
  if (dialect_->IsAdd(op)) {
    auto attr = op->getAttrOfType<TypeAttr>("T");
    return !attr || !mlir::isa<StringType>(attr.getValue());
  }
  const machina::OpDef *op_def = nullptr;
  absl::Status status = machina::OpRegistry::Global()->LookUpOpDef(
      op->getName().stripDialect().data(), &op_def);
  return status.ok() && op_def->is_aggregate();
}

bool OpCatHelper::IsCommutative(TFOp op) {
  if (dialect_->IsAdd(op)) {
    auto attr = op->getAttrOfType<TypeAttr>("T");
    return !attr || !mlir::isa<StringType>(attr.getValue());
  }
  const machina::OpDef *op_def = nullptr;
  absl::Status status = machina::OpRegistry::Global()->LookUpOpDef(
      op->getName().stripDialect().data(), &op_def);
  return status.ok() && op_def->is_commutative();
}

bool OpCatHelper::IsOnes(TFOp op) {
  if (dialect_->IsOnesLike(op)) return true;
  if (dialect_->IsZerosLike(op)) return false;

  if (dialect_->IsFill(op)) {
    TFOp value_op = op->getOperand(1).getDefiningOp();
    return !value_op && IsOnes(value_op);
  }

  if (!dialect_->IsConstant(op)) return false;

  SplatElementsAttr const_attr = op->getAttrOfType<SplatElementsAttr>("value");
  if (!const_attr) return false;

  return SplatElementsAttrHasValue(const_attr, 1);
}

bool OpCatHelper::IsZeros(TFOp op) {
  if (dialect_->IsOnesLike(op)) return false;
  if (dialect_->IsZerosLike(op)) return true;

  if (dialect_->IsFill(op)) {
    TFOp value_op = op->getOperand(1).getDefiningOp();
    return !value_op && IsZeros(value_op);
  }

  if (!dialect_->IsConstant(op)) return false;

  SplatElementsAttr const_attr = op->getAttrOfType<SplatElementsAttr>("value");
  if (!const_attr) return false;

  return SplatElementsAttrHasValue(const_attr, 0);
}

bool OpCatHelper::IsPersistent(TFOp op) {
  return dialect_->IsConstant(op) || dialect_->IsVariable(op) ||
         dialect_->IsHostConstant(op);
}

bool OpCatHelper::IsDataset(TFOp op) {
  static StringRef iterator_get_next = "IteratorGetNext";
  static StringRef iterator_get_next_sync = "IteratorGetNextSync";
  static StringRef dataset_to_single_element = "DatasetToSingleElement";
  static StringRef reduce_data_set = "ReduceDataset";
  StringRef op_name = op->getName().stripDialect();
  // See `GetNodeClassForOp` in core/graph/graph.cc.
  return op_name == iterator_get_next || op_name == iterator_get_next_sync ||
         op_name == dataset_to_single_element || op_name == reduce_data_set;
}

}  // namespace tfg
}  // namespace mlir
