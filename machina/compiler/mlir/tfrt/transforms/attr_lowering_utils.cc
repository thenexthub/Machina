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
#include "machina/compiler/mlir/tfrt/transforms/attr_lowering_utils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace machina {
namespace {

mlir::TypeAttr ConvertTypeAttribute(mlir::TypeAttr type_attr,
                                    mlir::Builder& builder) {
  auto type = type_attr.getValue();

  if (IsSupportedTfrtNumericDType(type)) return type_attr;

  // For TF custom types, we convert it to custom corert types.
  if (mlir::isa<mlir::TF::StringType>(type))
    return mlir::TypeAttr::get(
        tfrt::corert::StringType::get(builder.getContext()));

  if (mlir::isa<mlir::TF::ResourceType>(type))
    return mlir::TypeAttr::get(
        tfrt::corert::ResourceType::get(builder.getContext()));

  if (mlir::isa<mlir::TF::VariantType>(type))
    return mlir::TypeAttr::get(
        tfrt::corert::VariantType::get(builder.getContext()));

  if (mlir::isa<mlir::TF::Quint8Type>(type)) {
    return mlir::TypeAttr::get(
        tfrt::corert::Quint8Type::get(builder.getContext()));
  }

  if (mlir::isa<mlir::TF::Quint16Type>(type)) {
    return mlir::TypeAttr::get(
        tfrt::corert::Quint16Type::get(builder.getContext()));
  }

  if (mlir::isa<mlir::TF::Qint8Type>(type)) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint8Type::get(builder.getContext()));
  }

  if (mlir::isa<mlir::TF::Qint16Type>(type)) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint16Type::get(builder.getContext()));
  }

  if (mlir::isa<mlir::TF::Qint32Type>(type)) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint32Type::get(builder.getContext()));
  }

  // Return invalid results to emit error for unsupported types.
  return {};
}

mlir::Attribute ConvertAttribute(mlir::Attribute attr, mlir::Builder& builder) {
  // The supported attributes here should be kept consistent with
  // //third_party/tf_runtime/include/tfrt/core_runtime/op_attr_type.h
  //
  // Currently, not all machina data types are supported. Unranked shape
  // attributes are not supported yet.

  // Return directly if the attribute is already supported.
  if (mlir::isa<mlir::IntegerAttr, mlir::FloatAttr, mlir::BoolAttr,
                mlir::StringAttr, mlir::DenseIntOrFPElementsAttr>(attr))
    return attr;

  // For type attributes, we convert non-standard MLIR types to corresponding
  // corert types.
  if (auto type_attr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    if (auto shape_type =
            mlir::dyn_cast<mlir::TensorType>(type_attr.getValue())) {
      if (!shape_type.hasRank())
        return tfrt::corert::ShapeAttr::get(builder.getContext());

      return tfrt::corert::ShapeAttr::get(builder.getContext(),
                                          shape_type.getShape());
    }

    return ConvertTypeAttribute(type_attr, builder);
  }

  // Convert the attribute to the corresponding format in TFRT dialect if
  // needed.
  if (auto shape_attr = mlir::dyn_cast<mlir::TF::ShapeAttr>(attr)) {
    if (!shape_attr.hasRank())
      return tfrt::corert::ShapeAttr::get(builder.getContext());
    return tfrt::corert::ShapeAttr::get(builder.getContext(),
                                        shape_attr.getShape());
  }

  // For arrays, we recursively convert the elements.
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    toolchain::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(array_attr.size());
    for (auto attr : array_attr) {
      auto converted = ConvertAttribute(attr, builder);
      if (!converted) return {};
      attrs.push_back(converted);
    }
    return builder.getArrayAttr(attrs);
  }

  return {};
}

}  // namespace

bool IsSupportedTfrtNumericDType(mlir::Type type) {
  // Most of the machina data types (eg. f32, i64) are supported and they
  // are standard MLIR types that need no conversion here.
  if (type.isBF16() || type.isF16() || type.isF32() || type.isF64() ||
      type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
      type.isInteger(32) || type.isInteger(64) || type.isUnsignedInteger(8) ||
      type.isUnsignedInteger(16) || type.isUnsignedInteger(32) ||
      type.isUnsignedInteger(64))
    return true;

  if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32() || element_type.isF64()) return true;
  }

  return false;
}

// TODO(chky): attributes "_output_shapes" should be removed by any tool that
// generates TF MLIR dialect, as they are not used by CoreRuntime. Remove this
// filtering logic once unused attributes are cleaned up in the upper layer.
bool IsUnusedTfrtAttribute(toolchain::StringRef name) {
  // NOTE: attributes "f.*" are function attribute related and
  // are added during importing graph to MLIR TF Executor dialect. These
  // attributes are not actually used by TF ops with function attributes.
  // TODO(b/180399811): Re-evaluate the usage of these attributes.
  static const char* const kUnusedAttributes[] = {
      "_output_shapes",
      "resultSegmentSizes",
      "operandSegmentSizes",
  };

  for (auto attr : kUnusedAttributes) {
    if (name == attr) {
      return true;
    }
  }

  return name.contains("f.");
}

mlir::ArrayAttr CreateTfrtOpAttrs(toolchain::ArrayRef<mlir::NamedAttribute> attrs,
                                  mlir::Builder& builder) {
  toolchain::SmallVector<mlir::Attribute, 4> attr_array;
  for (auto key_and_value : attrs) {
    if (!IsUnusedTfrtAttribute(key_and_value.getName())) {
      auto converted = ConvertAttribute(key_and_value.getValue(), builder);
      if (!converted) return {};

      mlir::StringAttr key =
          builder.getStringAttr(key_and_value.getName().strref());
      attr_array.push_back(builder.getArrayAttr({key, converted}));
    }
  }
  return builder.getArrayAttr(attr_array);
}

}  // namespace machina
