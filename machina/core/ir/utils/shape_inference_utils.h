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

#ifndef MACHINA_CORE_IR_UTILS_SHAPE_INFERENCE_UTILS_H_
#define MACHINA_CORE_IR_UTILS_SHAPE_INFERENCE_UTILS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/platform/status.h"

namespace machina {
struct OpRegistrationData;
}  // namespace machina

namespace mlir {
namespace tfg {

// Function that takes in a value and extracts a constant from it, if available.
// If the value cannot be resolved as a constant, a nullptr will be returned.
// Certain shape functions require constant values as arguments.
using OperandAsConstantFn = toolchain::function_ref<Attribute(Value)>;

// Function that takes in an operation result and computes a shape (can be
// partial) value. Certain shape functions require shape values as arguments.
using OpResultAsShapeFn =
    toolchain::function_ref<machina::shape_inference::ShapeHandle(
        machina::shape_inference::InferenceContext&, OpResult)>;

// Function that takes a result index and returns the element type. Element
// types are necessary for handle types (resource, variant).
using ResultElementTypeFn = toolchain::function_ref<Type(int)>;

// Extracts the attributes of a MLIR operation and populates the converted
// attributes in a proto map<string, AttrValue>. This is used by operation
// defined in TF dialect which has different attributes format than TFG dialect.
using GetAttrValuesFn = toolchain::function_ref<absl::Status(
    Operation*, toolchain::StringRef, const machina::OpRegistrationData*, bool,
    machina::AttrValueMap*)>;

// Runs TensorFlow shape inference associated to the op type registered in the
// TensorFlow op registry based on the Graph version, operands, and attributes.
// Invoking this shape function will create conversions of parameters to the
// TensorFlow Graph equivalent data structures and back to MLIR equivalent data
// structures. This does not use a natively implemented shape inference in MLIR,
// and instead is temporary until shape functions are reimplemented/migrated to
// being in MLIR instead of the TensorFlow op registry.
// Note that the default way to get the attrs in the operation is using the API
// in TFG importer. For operations that has different format of attributes, they
// should give the `get_attr_values_fn` to read the attributes correctly.
LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, ValueRange operands,
    int64_t graph_version, OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    GetAttrValuesFn get_attr_values_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes);

// This one is almost the same as the above one, the difference is that we use
// ConvertOperationToNode to convert the operation to NodeDef to get the attr
// values.
LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, ValueRange operands,
    int64_t graph_version, OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_UTILS_SHAPE_INFERENCE_UTILS_H_
