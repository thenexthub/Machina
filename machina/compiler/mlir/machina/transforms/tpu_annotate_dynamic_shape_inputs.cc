/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "tf-tpu-annotate-dynamic-shape-inputs"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUANNOTATEDYNAMICSHAPEINPUTSPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class TPUAnnotateDynamicShapeInputsPass
    : public impl::TPUAnnotateDynamicShapeInputsPassBase<
          TPUAnnotateDynamicShapeInputsPass> {
  void runOnOperation() override;
};

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
  if (auto block_arg = mlir::dyn_cast<BlockArgument>(value))
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

void TPUAnnotateDynamicShapeInputsPass::runOnOperation() {
  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func_op) {
    Builder builder(cluster_func_op->getContext());
    // Skip non-tpu device cluster_func.
    auto cluster_id =
        cluster_func_op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
    if (!cluster_id) return WalkResult::advance();

    toolchain::SmallVector<int, 4> dynamic_shape_arg_index;

    // Traverse the operands of the cluster func op and find which operand
    // is returned by TPUAnnotateTensorsWithDynamicShapeOp.
    for (const auto& cluster_func_operand :
         toolchain::enumerate(cluster_func_op.getOperands())) {
      auto device_launch_op = toolchain::dyn_cast<tf_device::LaunchOp>(
          GetOpOfValue(cluster_func_operand.value()));
      if (!device_launch_op) continue;
      for (auto result : toolchain::zip(
               device_launch_op.getResults(),
               device_launch_op.GetBody().getTerminator()->getOperands())) {
        if (std::get<0>(result) == cluster_func_operand.value() &&
            toolchain::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(
                std::get<1>(result).getDefiningOp())) {
          dynamic_shape_arg_index.push_back(cluster_func_operand.index());
        }
      }
    }

    cluster_func_op->setAttr(TF::kDynamicArgIndexAttr,
                             builder.getI32ArrayAttr(dynamic_shape_arg_index));

    FlatSymbolRefAttr func_attr = cluster_func_op.getFuncAttr();
    func::FuncOp func =
        cluster_func_op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
            func_attr.getValue());

    // Update the marked argument with dynamic shapes.
    for (int index : dynamic_shape_arg_index) {
      BlockArgument arg = func.getArgument(index);
      auto inputType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      // Only rank 1 tensor is supported for now.
      if (!inputType || inputType.getRank() != 1) continue;
      auto shape = toolchain::to_vector<4>(inputType.getShape());
      toolchain::SmallVector<int64_t, 4> bounds(shape.begin(), shape.end());
      // Mark the dim as dynamic dim.
      shape[0] = ShapedType::kDynamic;
      auto extensions =
          mhlo::TypeExtensionsAttr::get(func->getContext(), bounds);
      auto resultType =
          RankedTensorType::get(shape, inputType.getElementType(), extensions);
      arg.setType(resultType);
    }
    toolchain::SmallVector<Type, 8> arg_types;
    for (auto arg : func.getArguments()) arg_types.push_back(arg.getType());
    func.setType(
        FunctionType::get(func.getContext(), arg_types,
                          func.front().getTerminator()->getOperandTypes()));
    return WalkResult::advance();
  });

  // Remove the annotated op after since it is just a placeholder.
  DenseSet<tf_device::LaunchOp> launch_ops;
  getOperation().walk([&](Operation* op) {
    if (toolchain::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(op)) {
      for (auto result : toolchain::zip(op->getOperands(), op->getResults())) {
        std::get<1>(result).replaceAllUsesWith(std::get<0>(result));
      }
      launch_ops.insert(op->getParentOfType<tf_device::LaunchOp>());
      op->erase();
    }
    return WalkResult::advance();
  });

  for (auto launch_op : launch_ops) {
    Block& block = launch_op.GetBody();
    if (&block.front() == &block.back()) {
      // The tf_device.launch is empty (except for the return).
      // Remove the whole tf_device.launch, since later passes will make it send
      // the arguments back and forth between the devices.
      Operation* return_op = &block.back();
      assert(toolchain::isa<tf_device::ReturnOp>(return_op));
      for (auto [inner, outer] :
           toolchain::zip(return_op->getOperands(), launch_op->getResults())) {
        outer.replaceAllUsesWith(inner);
      }
      launch_op->erase();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUAnnotateDynamicShapeInputsPass() {
  return std::make_unique<TPUAnnotateDynamicShapeInputsPass>();
}
}  // namespace TFTPU
}  // namespace mlir
