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

#include "machina/dtensor/mlir/expansions/sparse_to_dense_spmd_expander.h"

#include <optional>

#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> SparseToDenseSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Set the op's shape as the local shape of the input tensors from the
  // layouts.
  TF_ASSIGN_OR_RETURN(std::optional<Layout> computed_layout,
                      ExtractSingleLayoutFromOp(op));
  auto local_shape = computed_layout->LocalShapeFromGlobalShape(
      ExtractGlobalOutputShape(op->getResult(0)).value());
  auto op_result = op->getResult(0);

  const auto element_type =
      mlir::cast<mlir::TensorType>(op_result.getType()).getElementType();
  op_result.setType(mlir::RankedTensorType::get(local_shape, element_type));
  // No-op
  return op;
}

StatusOr<toolchain::DenseMap<int, Layout>>
SparseToDenseSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<toolchain::DenseMap<int, Layout>>
SparseToDenseSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return toolchain::DenseMap<int, Layout>();

  Layout output_layout = output_layouts.lookup(0);
  if (output_layout.mesh().is_tpu_mesh()) {
    return errors::InvalidArgument(
        "Layout for SparseToDenseOp must not be on TPU Mesh.");
  }
  return toolchain::DenseMap<int, Layout>({{0, output_layout}});
}

}  // namespace dtensor
}  // namespace machina
