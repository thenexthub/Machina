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

#ifndef MACHINA_DTENSOR_MLIR_VALUE_UTILS_H_
#define MACHINA_DTENSOR_MLIR_VALUE_UTILS_H_

#include "absl/status/status.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"
#include "machina/dtensor/cc/dstatus.h"

namespace machina {
namespace dtensor {

int ValueRank(mlir::Value operand_value);

// Creates a effective scalar type as rank 1 with a single element.
mlir::RankedTensorType EffectivelyScalarR1Type(mlir::Type element_type);

// Reshapes a value of size type tensor<i32> to scalar.
mlir::Value ReshapeSizeTypeToScalar(mlir::OpBuilder builder, mlir::Location loc,
                                    mlir::Value tensor);

// Retuns a int64 array representing the TensorFlow shape given the MLIR type.
// If the type is a resource, returns the underlying shape of the resource
// instead. Returns an error if the type is not a RankedTensorType.
StatusOr<toolchain::SmallVector<int64_t>> GetTFShapeFromType(mlir::Type type);

// Return a 1-D int32 constant array with the given values.
mlir::Value IntConst(mlir::OpBuilder& builder, mlir::Location loc,
                     toolchain::ArrayRef<int32> values);
// Return a 1-D int64 constant array with the given values.
mlir::Value Int64Const(mlir::OpBuilder& builder, mlir::Location loc,
                       toolchain::ArrayRef<int64_t> values);
// Return a 1-D float32 constant array with the given values.
mlir::Value FloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                       toolchain::ArrayRef<float> values);
// Returns a 1-D tf.string constant array with given values.
mlir::Value StringConst(mlir::OpBuilder& builder, mlir::Location loc,
                        toolchain::ArrayRef<toolchain::StringRef> values);
// Returns a tf.string scalar constant with given value.
mlir::Value StringScalarConst(mlir::OpBuilder& builder, mlir::Location loc,
                              toolchain::StringRef value);
// Returns a Int constant with the matching type.
mlir::Value IntConstWithMatchingType(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     toolchain::ArrayRef<int64_t> values,
                                     mlir::Type type);

StatusOr<int64_t> ExtractConstIntFromValue(mlir::Value value);
absl::Status ExtractConstVectorFromValue(
    mlir::Value value, toolchain::SmallVector<int64_t, 4>* out_vector);

// Returns a int64 scalar constant with `value`.
mlir::Value CreateIntScalarConst(int64_t value, mlir::OpBuilder builder,
                                 mlir::Location loc, bool use_int64 = true);

// Returns a scalar constant with 'value' of 'type'.
StatusOr<mlir::Value> CreateZeroScalarConst(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::Type type);

// Selects a scalar tensor value from a 1D array in specified index.
StatusOr<mlir::Value> SelectScalarValueFromArray(mlir::OpBuilder& builder,
                                                 int index,
                                                 mlir::Location location,
                                                 mlir::Value array);

// Returns the type that value holds. If value holds a Type that has a subtype,
// then it returns the subtype.
mlir::Type GetSubtypeOrSelf(mlir::Value value);

// Returns whether `val` is of resource type.
bool IsResourceType(mlir::Value val);

}  // namespace dtensor
}  // namespace machina
#endif  // MACHINA_DTENSOR_MLIR_VALUE_UTILS_H_
