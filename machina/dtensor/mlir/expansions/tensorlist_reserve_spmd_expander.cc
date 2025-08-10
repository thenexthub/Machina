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

#include "machina/dtensor/mlir/expansions/tensorlist_reserve_spmd_expander.h"

#include <cstdint>
#include <vector>

#include "toolchain/Support/Casting.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_location.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> TensorListReserveSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Recompute the new local shape from the Layout and the global shape.
  TF_ASSIGN_OR_RETURN(auto global_shape, GetShapeOfValue(op->getOpResult(0)));
  TF_ASSIGN_OR_RETURN(Layout layout, ExtractRequiredSingleLayoutFromOp(op));
  std::vector<int64_t> local_shape =
      layout.LocalShapeFromGlobalShape(global_shape);

  mlir::TF::TensorListReserveOp tensorlist_op =
      toolchain::dyn_cast<mlir::TF::TensorListReserveOp>(op);
  mlir::OpBuilder builder(op);

  mlir::Type element_type =
      mlir::cast<mlir::TensorType>(GetSubtypeOrSelf(op->getOpResult(0)))
          .getElementType();

  mlir::RankedTensorType new_output_type = mlir::RankedTensorType::get(
      {}, mlir::TF::VariantType::get(
              mlir::RankedTensorType::get(local_shape, element_type),
              builder.getContext()));
  mlir::Value new_shape_value = Int64Const(builder, DT_LOC(op), local_shape);
  mlir::TF::TensorListReserveOp new_op =
      builder.create<mlir::TF::TensorListReserveOp>(
          DT_LOC(op), new_output_type, new_shape_value,
          tensorlist_op.getNumElements());

  op->getResult(0).replaceAllUsesWith(new_op.getResult());
  op->erase();
  return new_op.getOperation();
}

// Always prefer the output to be replicated.
StatusOr<toolchain::DenseMap<int, Layout>>
TensorListReserveSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(auto global_shape, GetShapeOfValue(op->getOpResult(0)));
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, global_shape.size())}});
}

StatusOr<toolchain::DenseMap<int, Layout>>
TensorListReserveSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  // Do not infer any operand layouts.
  return toolchain::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace machina
