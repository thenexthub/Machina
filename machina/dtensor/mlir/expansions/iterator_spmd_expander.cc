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

#include "machina/dtensor/mlir/expansions/iterator_spmd_expander.h"

#include <cstdint>
#include <vector>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_location.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> IteratorGetNextSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::TF::IteratorGetNextOp original_op =
      mlir::cast<mlir::TF::IteratorGetNextOp>(op);
  mlir::OpBuilder builder(op);

  TF_ASSIGN_OR_RETURN(std::vector<Layout> output_layouts,
                      ExtractRequiredLayoutFromOp(op));

  toolchain::SmallVector<mlir::Type, 4> local_types(original_op->getNumResults());

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    mlir::TensorType global_output_type =
        mlir::cast<mlir::TensorType>(original_op.getResult(i).getType());
    std::vector<int64_t> local_shape =
        output_layouts[i].LocalShapeFromGlobalShape(
            global_output_type.getShape());
    local_types[i] = mlir::RankedTensorType::get(
        local_shape, global_output_type.getElementType());
  }

  auto new_op = builder.create<mlir::TF::IteratorGetNextOp>(
      DT_LOC(op->getLoc()), local_types, original_op->getOperand(0));

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    original_op.getResult(i).replaceAllUsesWith(new_op.getResult(i));
  }
  original_op.erase();
  return InferSPMDExpandedLocalShape(new_op);
}

StatusOr<toolchain::DenseMap<int, Layout>>
IteratorGetNextSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // Extract the output element layouts from the `tf._element_layouts` attribute
  // of the iterator resource tensor.
  TF_ASSIGN_OR_RETURN(const auto layouts,
                      ExtractElementLayoutsFromOperand(op->getOpOperand(0)));

  toolchain::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] = layouts[i];
  }
  return output_layouts;
}

StatusOr<toolchain::DenseMap<int, Layout>>
IteratorGetNextSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Iterator resource tensors are always 0-dimensional.
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<mlir::Operation*> IteratorGetNextAsOptionalSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Extract the output element layouts from the `tf._element_layouts` attribute
  // of the iterator resource tensor.
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractElementLayoutsFromOperand(op->getOpOperand(0)));

  auto array_attr = op->getAttrOfType<mlir::ArrayAttr>(kIteratorOutputShapes);
  if (!array_attr)
    return errors::InvalidArgument(
        toolchain::formatv("Could not find `{0}` attribute of op: {1}",
                      kIteratorOutputShapes, op->getName())
            .str());

  toolchain::SmallVector<mlir::Attribute, 4> output_shape_attrs(array_attr.size());
  for (int i = 0; i < array_attr.size(); ++i) {
    std::vector<int64_t> local_shape =
        output_layouts[i].LocalShapeFromGlobalShape(
            mlir::cast<mlir::TF::ShapeAttr>(array_attr[i]).getShape());
    output_shape_attrs[i] = mlir::cast<mlir::Attribute>(
        mlir::TF::ShapeAttr::get(op->getContext(), {local_shape}));
  }

  // Update the `output_shapes` attribute on the op to match the local shape
  // based on the iterator element layouts.
  op->setAttr(kIteratorOutputShapes,
              mlir::ArrayAttr::get(op->getContext(), output_shape_attrs));
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<toolchain::DenseMap<int, Layout>>
IteratorGetNextAsOptionalSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Variant tensors are always 0-dimensional.
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<toolchain::DenseMap<int, Layout>>
IteratorGetNextAsOptionalSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Iterator resource tensors are always 0-dimensional.
  return toolchain::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

}  // namespace dtensor
}  // namespace machina
