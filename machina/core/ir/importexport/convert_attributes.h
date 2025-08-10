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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/resource_handle.pb.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/platform/statusor.h"

namespace mlir {
namespace tfg {

// Convert the list of MLIR Attributes `attrs` to the `machina::AttrValueMap`
// `values`.
absl::Status ConvertAttributes(ArrayRef<NamedAttribute> attrs,
                               ArrayRef<StringRef> attrs_to_ignore,
                               bool remove_ref_type,
                               machina::AttrValueMap* values);

// Convert the MLIR attribute `attr` and return a `machina::AttrValue`.
absl::StatusOr<machina::AttrValue> ConvertAttribute(Attribute attr);

absl::Status SetShapeAttribute(absl::string_view name, ShapedType shaped_type,
                               machina::AttrValueMap* values);

// Converts an MLIR shaped type to a TensorFlow shape attribute.
ShapeAttr ConvertTypeToTensorShapeAttr(const Type& type);

/// Import from TensorFlow to MLIR

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
absl::StatusOr<Attribute> ConvertNonFuncAttributeValue(
    const machina::AttrValue& value, Builder& builder);

// Converts all kinds of AttrValue proto into an MLIR attribute.
absl::StatusOr<Attribute> ConvertAttributeValue(
    const machina::AttrValue& value, Builder& builder);

// Convert the MLIR FullTyoe attribute `attr` and return a
// `machina::FullTypeDef`.
absl::StatusOr<machina::FullTypeDef> ConvertAttribute(
    tf_type::FullTypeAttr full_type);

// Converts fulltype proto to attribute.
absl::StatusOr< ::mlir::tf_type::FullTypeAttr> ConvertAttribute(
    const machina::FullTypeDef& full_type, Builder& builder);

// Convert an array of handle data (pairs of data types and shapes) to an array
// attribute of tensor types.
absl::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder,
    const machina::protobuf::RepeatedPtrField<
        machina::ResourceHandleProto_DtypeAndShape>& handle_data);

// Convert an array of handle data into the `handle_data` field of the provided
// ArgDef. Each entry of the array is expected to be a TensorType.
absl::Status ConvertHandleData(ArrayAttr handle_data_arr,
                               machina::OpDef::ArgDef* arg);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_CONVERT_ATTRIBUTES_H_
