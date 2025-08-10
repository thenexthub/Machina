/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_OPTIMIZE_BATCH_MATMUL_UTILS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_OPTIMIZE_BATCH_MATMUL_UTILS_H_

#include <cstdint>
#include <tuple>
#include <utility>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {

// LHS and RHS of BatchMatMulOp has shapes following the pattern:
// B0,...,Bn,L,C and B0,...,Bn,C,R. The output shape of BatchMatMulOp is:
// B0,...,Bn,L,R.
//
// LHS and RHS of FullyConnectedOp has shapes following the pattern:
// B0,...,Bn,L,C and R,C. The output shape of FullyConnectedOp is:
// B0,...,Bn,L,R.
//
// The fundamental idea behind seeing transposes and reshapes around
// BatchMatMulOp is that-
// -- BatchMatMulOp is often created as a result of lowering einsum or
//    dot_general ops.
// -- einsum and dot_general ops have multiple contracting and output
//    dimensions that will to be reshaped and transposed to match the
//    BatchMatMulOp's LHS and RHS restrictions.
//
// This file contains utility functions to identify the reshapes and transposes
// around BatchMatMulOp and see if they can be fused.

// A struct to hold axes and sizes for a set of dimensions.
struct DimensionVector {
  toolchain::ArrayRef<int64_t> AxesArray() const { return axes; }
  toolchain::ArrayRef<int64_t> SizesArray() const { return sizes; }

  toolchain::SmallVector<int64_t, 4> axes;
  toolchain::SmallVector<int64_t, 4> sizes;
};

// A struct to hold information about dimensions of dot_general operands.
class BatchMatMulDimensionsInfo {
 public:
  BatchMatMulDimensionsInfo(mlir::ShapedType type, bool is_lhs);
  const DimensionVector& batch_dimensions() const;
  const DimensionVector& contracting_dimensions() const;
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  const DimensionVector& out_dimensions() const;
  bool is_lhs() const;

 private:
  DimensionVector batch_dimensions_;
  DimensionVector contracting_dimensions_;
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  DimensionVector out_dimensions_;
  bool is_lhs_;
};

// Returns the dimensions info of the LHS of BatchMatMulOp.
BatchMatMulDimensionsInfo GetBatchMatMulLhsDimensionsInfo(
    mlir::ShapedType type);

// Returns the dimensions info of the RHS of BatchMatMulOp.
BatchMatMulDimensionsInfo GetBatchMatMulRhsDimensionsInfo(
    mlir::ShapedType type);

// Returns true if the product of the last few dimensions in the
// `reshape_input_shape` is equal to the contracting dimension of the
// `bmm_dimensions_info`.
bool HasFlattenedContractingDims(
    toolchain::ArrayRef<int32_t> reshape_input_shape,
    const BatchMatMulDimensionsInfo& bmm_dimensions_info);

// Returns true if the product of the first few dimensions in the
// `reshape_input_shape` is equal to the output dimension of the
// `bmm_dimensions_info`.
bool HasFlattenedOutDims(toolchain::ArrayRef<int32_t> reshape_input_shape,
                         const BatchMatMulDimensionsInfo& bmm_dimensions_info);

// Returns true if the contracting and output dimensions are transposed in the
// `transpose_permutation`.
bool HasTransposedContractingAndOutDims(
    toolchain::ArrayRef<int32_t> transpose_input_shape,
    toolchain::ArrayRef<int32_t> transpose_permutation,
    const BatchMatMulDimensionsInfo& bmm_dimensions_info);

// `transpose_permutation` is the permutation of the input shape of the
// transpose op. `transpose_input_shape` is the shape of the input of the
// transpose op. `bmm_dimensions_info` is the dimensions info of the
// BatchMatMulOp.
//
// The dimensions in the transpose_permutation can be split into three groups:
//   1. Batch dimensions
//   2. Contracting dimensions
//   3. Output dimensions
//
// - The number of dimensions and the order of the dimensions in the
//   batch-dimensions group is expected to match the batch dimensions of the
//   BatchMatMulOp.
// - The number of dimensions in the contracting-dimensions and
//   output-dimensions groups can be more than 1.
// - The dimensions in group 1 are expected to be a monotonically increasing
//   sequence.
// - The dimensions in group 2 and 3 need not be a monotonically increasing
//   sequence.
// - In this function, we only care if the groups 2 and 3 are transposed.
//
// For example, consider the following transpose_permutation-
// [0, 1, 2, 6, 7, 8, 3, 4, 5]. Here all the three groups are monotonically
// increasing. But other permutations like [0, 1, 2, 8, 7, 6, 4, 5, 3] and [0,
// 1, 2, 6, 7, 8, 3, 5, 4] are also valid.
//
// NOTE: The first version of this function will support the case where all the
// three groups are monotonically increasing.
std::tuple<std::pair<int, int>, std::pair<int, int>>
GetTransposedGroupsIndexRange(toolchain::ArrayRef<int32_t> transpose_permutation);

}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_OPTIMIZE_BATCH_MATMUL_UTILS_H_
