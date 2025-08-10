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

#include <memory>
#include <string>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORPROPAGATEDEFAULTLAYOUT
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

// Rewrites Op to a tf.DTensorLayout op that forwards `input` value.
void CreateDTensorLayoutOp(const Layout& layout, mlir::Value input,
                           mlir::TensorType& type, mlir::Location loc,
                           mlir::IntegerAttr arg_index,
                           mlir::OpBuilder* builder,
                           mlir::MLIRContext* context) {
  if (layout.IsEmpty()) return;

  auto layout_op = builder->create<mlir::TF::DTensorLayout>(
      loc, input, mlir::dtensor::LayoutAttr::get(context, layout),
      mlir::TF::ShapeAttr::get(context, type));
  if (arg_index != nullptr) {
    layout_op->setAttr(kFromArgIndex, arg_index);
  }
  toolchain::SmallPtrSet<mlir::Operation*, 4> exception{layout_op};
  input.replaceAllUsesExcept(layout_op.getOutput(), exception);
}

// Adds DTensorLayout op following each Relayout operation to ensure that
// tensor from `relayout` has fixed layout.
mlir::LogicalResult PropagateDTensorLayoutForRelayout(
    mlir::MLIRContext& c, mlir::TF::RelayoutOp relayout) {
  const std::string layout_str = relayout.getLayout().str();
  auto layout_or_status = Layout::FromString(layout_str);
  if (!layout_or_status.ok()) {
    return relayout.emitOpError(
        toolchain::formatv("found Relayout op with incorrect/unparsable layout. "
                      "Found layout: {0} ",
                      layout_str));
  }
  const Layout& layout = layout_or_status.value();

  // Skip adding a DTensorLayout if Relayout is 'dynamic'. Any dimension with
  // MATCH for the layout will have its layout preserved in layout propagation.
  for (const std::string& sharding_spec : layout.sharding_spec_strs())
    if (sharding_spec == Layout::kMatch) return mlir::success();

  mlir::OpBuilder builder(relayout->getBlock(),
                          ++mlir::Block::iterator(relayout));
  mlir::TensorType type = mlir::dyn_cast<mlir::TensorType>(relayout.getType());
  if (!type) return relayout.emitOpError("type required for Relayout op");

  CreateDTensorLayoutOp(layout, relayout.getOutput(), type, relayout.getLoc(),
                        nullptr, &builder, &c);
  return mlir::success();
}

// Creates tf.DTensorLayout that is connected to each function argument if
// function arg contains layout attribute.
mlir::LogicalResult PropagateFunctionArgAttrToLayoutOp(
    mlir::MLIRContext& c, mlir::func::FuncOp function) {
  for (int arg_index = 0; arg_index < function.getNumArguments(); ++arg_index) {
    auto layout_attr = function.getArgAttrOfType<mlir::StringAttr>(
        arg_index, kCustomDeviceAttr);
    if (!layout_attr) continue;
    const auto layout_str = layout_attr.getValue().str();
    auto layout_or_status = Layout::FromString(layout_str);
    if (!layout_or_status.ok())
      return function.emitOpError(toolchain::formatv(
          "function includes attribute {0} for {1}-th arg that cannot be "
          "serialized to correct layout format. Found attribute {3}",
          kCustomDeviceAttr, arg_index, layout_str));

    mlir::OpBuilder builder(function.getBody());
    auto arg = function.getArgument(arg_index);
    mlir::Type tensor_type = GetSubtypeOrSelf(arg);
    if (auto type = mlir::dyn_cast<mlir::TensorType>(tensor_type)) {
      CreateDTensorLayoutOp(layout_or_status.value(), arg, type,
                            function.getLoc(),
                            builder.getI64IntegerAttr(arg_index), &builder, &c);

    } else {
      return function.emitOpError()
             << "is missing tensor type for argument " << arg_index;
    }
  }

  return mlir::success();
}

// Creates tf.DTensorLayout that is connected to terminator op of function if
// function contains default layout attribute that represents layout of function
// outputs.
mlir::LogicalResult PropagateFunctionDefaultLayoutAttrToLayoutOp(
    mlir::MLIRContext& c, mlir::func::FuncOp function) {
  for (int ret_index = 0; ret_index < function.getNumResults(); ++ret_index) {
    auto layout_attr_from_func_result =
        function.getResultAttrOfType<mlir::StringAttr>(
            ret_index, kCustomDefaultLayoutAttr);
    if (!layout_attr_from_func_result) continue;

    const std::string layout_string =
        layout_attr_from_func_result.getValue().str();
    auto result_layout_or_status = Layout::FromString(layout_string);
    if (!result_layout_or_status.ok())
      return function.emitOpError(
          toolchain::formatv("function includes default layout attribute {0} for "
                        "{1}-th output that cannot be serialized to correct "
                        "layout format. Found attribute {3}",
                        kCustomDefaultLayoutAttr, ret_index, layout_string));

    auto function_terminator = function.getBody().front().getTerminator();
    mlir::OpBuilder builder(function_terminator);
    auto return_value = function_terminator->getOperand(ret_index);

    if (auto type = mlir::dyn_cast<mlir::TensorType>(return_value.getType()))
      CreateDTensorLayoutOp(result_layout_or_status.value(), return_value, type,
                            function.getLoc(), nullptr, &builder, &c);
    else
      return function.emitOpError()
             << "is missing tensor type for result " << ret_index;
  }

  return mlir::success();
}

mlir::LogicalResult PropagateOpAttrToLayoutOp(mlir::MLIRContext& context,
                                              mlir::func::FuncOp function) {
  auto walk_result =
      function.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (auto relayout = toolchain::dyn_cast<mlir::TF::RelayoutOp>(op)) {
          (void)PropagateDTensorLayoutForRelayout(context, relayout);
          return mlir::WalkResult::advance();
        }

        auto layout_or_status = ExtractLayoutFromOp(op);
        auto arg_index = op->getAttrOfType<mlir::IntegerAttr>(kFromArgIndex);
        if (!layout_or_status.ok()) {
          op->emitOpError(toolchain::formatv(
              "op has layout attribute {0} that cannot be deserizlied.",
              kLayoutAttr));
          return mlir::WalkResult::interrupt();
        }

        mlir::OpBuilder builder(&context);
        builder.setInsertionPointAfter(op);
        const auto layouts = layout_or_status.value();
        for (const auto& layout_and_index : toolchain::enumerate(layouts)) {
          const int index = layout_and_index.index();
          const auto& layout = layout_and_index.value();
          if (!layout || layout->IsEmpty()) continue;

          auto op_output = op->getResult(index);
          if (auto type =
                  mlir::dyn_cast<mlir::TensorType>(op_output.getType())) {
            CreateDTensorLayoutOp(*layout, op_output, type, function.getLoc(),
                                  arg_index, &builder, &context);
          } else {
            return op->emitOpError()
                   << "type for output " << index << " is not a TensorType";
          }
        }

        return mlir::WalkResult::advance();
      });
  if (walk_result.wasInterrupted()) return mlir::failure();
  return mlir::success();
}

// MLIR pass that removes trivially unused operations in graph.
struct DTensorPropagateDefaultLayout
    : public impl::DTensorPropagateDefaultLayoutBase<
          DTensorPropagateDefaultLayout> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto function = getOperation();

    // Set user annotated layout on operations.
    if (mlir::failed(PropagateOpAttrToLayoutOp(context, function)))
      return signalPassFailure();

    // Set user annotated layout on function arguments.
    if (mlir::failed(PropagateFunctionArgAttrToLayoutOp(context, function)))
      return signalPassFailure();

    // Set user annotated layout on function outputs.
    if (mlir::failed(
            PropagateFunctionDefaultLayoutAttrToLayoutOp(context, function)))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorPropagateDefaultLayout() {
  return std::make_unique<DTensorPropagateDefaultLayout>();
}

}  // namespace dtensor
}  // namespace machina
