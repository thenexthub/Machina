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

#ifndef MACHINA_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_
#define MACHINA_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_

#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace machina {
namespace dtensor {

// Gets the SparseToDenseOp that generates `value` if `value` is the result of
// a SparseToDenseOp. Returns empty otherwise. This is useful
// in SparseExpansion where we want to check whether some operand
// is a SparseTensor, by checking whether that operand is a result of a
// SparseToDenseOp. If this value is eventually an output of a SparseToDenseOp,
// there should only be DTensor related ops between the actual SparseToDenseOp,
// e.g. DTensorRelayout ops or DTensorLayout op.
absl::StatusOr<mlir::TF::SparseToDenseOp> GetSparseToDenseOp(mlir::Value value);

// Checks whether `value is an output of a SparseToDenseOp value.
bool IsSparseValue(mlir::Value value);

// Checks if `op` has any sparse value operands.
bool HasAnySparseInput(mlir::Operation* op);

// Checks if all operands of `op` is a sparse value.
bool AllSparseInput(mlir::Operation* op);

// Returns the indices component dense tensor from `value`. `value` represents
// a SparseTensor value.
absl::StatusOr<mlir::Value> GetIndicesFromSparseTensor(mlir::Value value);

// Returns the values component dense tensor from `value`.`value` represents
// a SparseTensor value.
absl::StatusOr<mlir::Value> GetValuesFromSparseTensor(mlir::Value value);

// Returns the dense shape component dense tensor from `value`. `value`
// represents a SparseTensor value.
absl::StatusOr<mlir::Value> GetDenseShapesFromSparseTensor(mlir::Value value);

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_
