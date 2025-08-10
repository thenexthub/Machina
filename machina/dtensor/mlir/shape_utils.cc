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

#include "machina/dtensor/mlir/shape_utils.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "toolchain/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/utils/shape_inference_utils.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/public/version.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

StatusOr<toolchain::ArrayRef<int64_t>> ExtractGlobalInputShape(
    mlir::OpOperand& input_value) {
  const int operand_index = input_value.getOperandNumber();
  auto input_defining_op = input_value.get().getDefiningOp();

  if (input_defining_op) {
    if (auto layout_op =
            toolchain::dyn_cast<mlir::TF::DTensorLayout>(input_defining_op)) {
      auto global_shape = layout_op.getGlobalShape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
    return ExtractGlobalOutputShape(cast<mlir::OpResult>(input_value.get()));
  }

  // If we reach this point, we're working with a function argument.
  auto op = input_value.getOwner();
  auto enclosing_function = op->getParentOfType<mlir::func::FuncOp>();
  if (!enclosing_function)
    return errors::InvalidArgument(
        toolchain::formatv("Could not find global shape of {0}-th input to op: {1}",
                      operand_index, op->getName())
            .str());

  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(input_value.get());
  auto global_shape_attr =
      enclosing_function.getArgAttrOfType<mlir::TF::ShapeAttr>(
          block_arg.getArgNumber(), kGlobalShapeDialectAttr);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`tf._global_shape` attribute of operation not found.");

  return global_shape_attr.getShape();
}

StatusOr<toolchain::ArrayRef<int64_t>> ExtractGlobalOutputShape(
    mlir::OpResult result_value) {
  auto op = result_value.getOwner();
  const int output_index = result_value.getResultNumber();

  if (op->getOpResult(output_index).hasOneUse()) {
    auto user = op->getOpResult(output_index).getUses().begin().getUser();
    if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(user)) {
      auto global_shape = layout_op.getGlobalShape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
  }

  auto global_shape_attr = op->getAttrOfType<mlir::ArrayAttr>(kGlobalShape);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`_global_shape` attribute of operation not found.");

  const int num_results = op->getNumResults();
  assert(global_shape_attr.size() == num_results);

  if (output_index >= op->getNumResults())
    return errors::InvalidArgument(
        toolchain::formatv("Requested global shape of {0} output but op has only "
                      "{1} return values.",
                      output_index, num_results)
            .str());

  auto shape_attr = global_shape_attr[output_index];
  return toolchain::cast<mlir::tf_type::ShapeAttr>(shape_attr).getShape();
}

namespace {

// Extracts attributes from a MLIR operation, including derived attributes, into
// one NamedAttrList.
mlir::NamedAttrList GetAllAttributesFromOperation(mlir::Operation* op) {
  mlir::NamedAttrList attr_list;
  attr_list.append(op->getAttrDictionary().getValue());

  if (auto derived = toolchain::dyn_cast<mlir::DerivedAttributeOpInterface>(op)) {
    auto materialized = derived.materializeDerivedAttributes();
    attr_list.append(materialized.getValue());
  }

  return attr_list;
}

// Infers output shape of `op` given its local operand shape. For shape
// inference function that requires input operation to be a constant, if input
// operation is `DTensorLayout` op, then we use input of DTensorLayout op
// instead for correct constant matching.
mlir::LogicalResult InferShapeOfTFOpWithCustomOperandConstantFn(
    std::optional<mlir::Location> location, mlir::Operation* op,
    int64_t graph_version,
    toolchain::SmallVectorImpl<mlir::ShapedTypeComponents>& inferred_return_shapes) {
  if (auto type_op = toolchain::dyn_cast<mlir::InferTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    toolchain::SmallVector<mlir::Type, 4> inferred_return_types;
    auto result = type_op.inferReturnTypes(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getPropertiesStorage(), op->getRegions(), inferred_return_types);
    if (failed(result)) return mlir::failure();

    inferred_return_shapes.resize(inferred_return_types.size());
    for (const auto& inferred_return_type :
         toolchain::enumerate(inferred_return_types)) {
      if (auto shaped_type =
              toolchain::dyn_cast<mlir::ShapedType>(inferred_return_type.value())) {
        if (shaped_type.hasRank()) {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getShape(),
                                         shaped_type.getElementType());
        } else {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getElementType());
        }
      }
    }

    return mlir::success();
  }

  if (auto shape_type_op =
          toolchain::dyn_cast<mlir::InferShapedTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    return shape_type_op.inferReturnTypeComponents(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getPropertiesStorage(), op->getRegions(), inferred_return_shapes);
  }

  // If `operand` is from DTensorLayout op, use input value of DTensorLayout op
  // instead.
  auto operand_as_constant_fn = [](mlir::Value operand) -> mlir::Attribute {
    while (auto input_op = toolchain::dyn_cast_or_null<mlir::TF::DTensorLayout>(
               operand.getDefiningOp())) {
      operand = input_op.getInput();
    }

    mlir::Attribute attr;
    if (matchPattern(operand, m_Constant(&attr))) return attr;
    return nullptr;
  };

  auto op_result_as_shape_fn =
      [](shape_inference::InferenceContext& ic,
         mlir::OpResult op_result) -> shape_inference::ShapeHandle {
    auto rt = toolchain::dyn_cast<mlir::RankedTensorType>(op_result.getType());
    if (!rt || rt.getRank() != 1 || !rt.hasStaticShape()) return {};

    std::vector<shape_inference::DimensionHandle> dims(rt.getDimSize(0),
                                                       ic.UnknownDim());
    mlir::Attribute attr;
    if (matchPattern(op_result, m_Constant(&attr))) {
      auto elements = toolchain::dyn_cast<mlir::DenseIntElementsAttr>(attr);
      if (elements)
        for (const auto& element :
             toolchain::enumerate(elements.getValues<toolchain::APInt>()))
          dims[element.index()] = ic.MakeDim(element.value().getSExtValue());
    }
    return ic.MakeShape(dims);
  };

  auto result_element_type_fn = [](int) -> mlir::Type { return nullptr; };

  return mlir::TF::InferReturnTypeComponentsForTFOp(
      location, op, graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn, inferred_return_shapes);
}

}  // namespace

absl::Status InferSPMDExpandedLocalShapeForResourceOutput(
    mlir::OpResult* op_result, const Layout& output_layout,
    mlir::MLIRContext* context) {
  if (toolchain::isa<mlir::TF::ResourceType>(
          mlir::getElementTypeOrSelf(*op_result))) {
    TF_ASSIGN_OR_RETURN(toolchain::ArrayRef<int64_t> global_shape,
                        GetGlobalShapeOfValueFromDTensorLayout(*op_result));
    const std::vector<int64_t>& local_shape =
        output_layout.LocalShapeFromGlobalShape(global_shape);
    auto resource_type = toolchain::dyn_cast<mlir::tf_type::ResourceType>(
        toolchain::cast<mlir::TensorType>(op_result->getType()).getElementType());

    auto sub_types = resource_type.getSubtypes();
    auto resource_arg_sub_type = sub_types.front();

    // The local shape that is to be assigned to this resource output.
    toolchain::SmallVector<int64_t, 4> local_arg_shape(local_shape.begin(),
                                                  local_shape.end());

    auto local_variable_subtype = mlir::RankedTensorType::get(
        local_arg_shape, resource_arg_sub_type.getElementType());
    auto new_var_type = mlir::RankedTensorType::get(
        {},
        mlir::TF::ResourceType::get(
            mlir::ArrayRef<mlir::TensorType>{local_variable_subtype}, context));
    op_result->setType(new_var_type);
  }
  return absl::OkStatus();
}

mlir::Operation* InferSPMDExpandedLocalShape(mlir::Operation* op) {
  toolchain::SmallVector<mlir::ShapedTypeComponents, 4> inferred_return_types;
  (void)InferShapeOfTFOpWithCustomOperandConstantFn(
      op->getLoc(), op, TF_GRAPH_DEF_VERSION, inferred_return_types);
  assert(inferred_return_types.size() == op->getNumResults());

  for (auto it : toolchain::zip(inferred_return_types, op->getOpResults())) {
    const auto& return_type = std::get<0>(it);
    auto& op_result = std::get<1>(it);
    const auto element_type =
        toolchain::cast<mlir::TensorType>(op_result.getType()).getElementType();

    if (return_type.hasRank()) {
      op_result.setType(
          mlir::RankedTensorType::get(return_type.getDims(), element_type));
    } else {
      op_result.setType(mlir::UnrankedTensorType::get(element_type));
    }
  }

  return op;
}

StatusOr<toolchain::ArrayRef<int64_t>> GetShapeOfValue(const mlir::Value& value,
                                                  bool fail_on_dynamic) {
  // Getting the subtype or self allows supporting extracting the underlying
  // shape that variant or resource tensors point to.
  mlir::Type type = GetSubtypeOrSelf(value);
  if (auto ranked_type = toolchain::dyn_cast<mlir::RankedTensorType>(type)) {
    if (ranked_type.hasStaticShape() || !fail_on_dynamic)
      return ranked_type.getShape();
    else
      return errors::InvalidArgument("value shape is not static");
  }
  return errors::InvalidArgument("value type is not a RankedTensorType");
}

StatusOr<toolchain::ArrayRef<int64_t>> GetGlobalShapeOfValueFromDTensorLayout(
    const mlir::Value& value) {
  if (mlir::isa<mlir::OpResult>(value) &&
      mlir::isa<mlir::TF::DTensorLayout>(value.getDefiningOp())) {
    auto layout_op = mlir::cast<mlir::TF::DTensorLayout>(value.getDefiningOp());
    if (layout_op.getGlobalShape()) return layout_op.getGlobalShape().value();
  } else if (value.hasOneUse() &&
             mlir::isa<mlir::TF::DTensorLayout>(*value.getUsers().begin())) {
    auto layout_op =
        mlir::cast<mlir::TF::DTensorLayout>(*value.getUsers().begin());
    if (layout_op.getGlobalShape()) return layout_op.getGlobalShape().value();
  }
  return errors::InvalidArgument(
      "consumer or producer of value is not a DTensorLayout");
}

}  // namespace dtensor
}  // namespace machina
