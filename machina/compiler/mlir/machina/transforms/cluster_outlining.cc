/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kFuncAttr[] = "func";

#define GEN_PASS_DEF_CLUSTEROUTLININGPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct ClusterOutliningPass
    : public impl::ClusterOutliningPassBase<ClusterOutliningPass> {
  void runOnOperation() override;
};

#define GEN_PASS_DEF_LAUNCHOUTLININGPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct LaunchOutliningPass
    : public impl::LaunchOutliningPassBase<LaunchOutliningPass> {
  void runOnOperation() override;
};

void ReplaceClusterReturnWithReturn(tf_device::ReturnOp cluster_return_op,
                                    OpBuilder* builder) {
  builder->create<func::ReturnOp>(cluster_return_op.getLoc(),
                                  cluster_return_op.getOperands());
  cluster_return_op.erase();
}

// Builds a function that outlines region attached to cluster_op or launch_op,
// and inserts built function into given module.
template <typename ClusterOrLaunchOp>
func::FuncOp BuildFunction(toolchain::ArrayRef<Value> live_ins, ClusterOrLaunchOp op,
                           SymbolTable* symbol_table, OpBuilder* builder) {
  toolchain::SmallVector<Type, 4> operand_types;
  operand_types.reserve(live_ins.size());
  for (Value v : live_ins) operand_types.emplace_back(v.getType());

  auto func_type = builder->getFunctionType(operand_types, op.getResultTypes());

  std::string func_name;
  if (auto outlined_func_name = op->template getAttrOfType<StringAttr>(
          TF::kClusterOutlinedFunctionNameAttr)) {
    op->removeAttr(TF::kClusterOutlinedFunctionNameAttr);
    func_name = outlined_func_name.str();
  } else {
    func_name = "_func";
  }

  func::FuncOp outlined_func =
      func::FuncOp::create(op.getLoc(), func_name, func_type);

  // This function is not externally visible and marking it private would allow
  // symbol-dce pass to remove it when it is not referenced anymore.
  outlined_func.setPrivate();

  // Create function body.
  Block* outlined_func_block = outlined_func.addEntryBlock();

  // Replace uses of live-in values within cluster_op region with function
  // arguments.
  Region& op_region = op.getBody();
  for (auto p : toolchain::zip(live_ins, outlined_func_block->getArguments())) {
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p), op_region);
  }

  // Move all instructions in cluster_op into outlined_function's only block.
  auto& op_body = op.GetBody().getOperations();
  outlined_func_block->getOperations().splice(
      outlined_func_block->end(), op_body, op_body.begin(), op_body.end());

  // Replace `tf_device.return` terminator with `std.return` in function
  // body.
  auto return_op =
      cast<tf_device::ReturnOp>(outlined_func_block->getTerminator());
  builder->setInsertionPoint(return_op);
  ReplaceClusterReturnWithReturn(return_op, builder);

  symbol_table->insert(outlined_func);
  return outlined_func;
}

// Outlines body of `tf_device.cluster` into a function and create a
// `tf_device.cluster_func` to invoke that function. `tf_device.cluster` is
// removed afterwards.`
void OutlineCluster(tf_device::ClusterOp cluster_op, SymbolTable* symbol_table,
                    OpBuilder* builder) {
  toolchain::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(cluster_op.getBody(), cluster_op.getBody(),
                            live_ins);

  func::FuncOp outlined_func =
      BuildFunction(live_ins.getArrayRef(), cluster_op, symbol_table, builder);
  cluster_op->setAttr(
      builder->getStringAttr(kFuncAttr),
      mlir::SymbolRefAttr::get(builder->getContext(), outlined_func.getName()));

  builder->setInsertionPoint(cluster_op);
  auto cluster_func_op = builder->create<tf_device::ClusterFuncOp>(
      cluster_op.getLoc(), outlined_func.getFunctionType().getResults(),
      live_ins.getArrayRef(), cluster_op->getAttrs());
  cluster_op.replaceAllUsesWith(cluster_func_op);
  cluster_op.erase();
}

// Outlines body of `tf_device.launch` into a function and create a
// `tf_device.launch_func` to invoke that function. `tf_device.launch` is
// removed afterwards.`
void OutlineLaunch(tf_device::LaunchOp launch_op, SymbolTable* symbol_table,
                   OpBuilder* builder) {
  toolchain::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(launch_op.getBody(), launch_op.getBody(), live_ins);

  func::FuncOp outlined_func =
      BuildFunction(live_ins.getArrayRef(), launch_op, symbol_table, builder);
  launch_op->setAttr(
      builder->getStringAttr(kFuncAttr),
      mlir::SymbolRefAttr::get(builder->getContext(), outlined_func.getName()));

  builder->setInsertionPoint(launch_op);
  auto cluster_func_op = builder->create<tf_device::LaunchFuncOp>(
      launch_op.getLoc(), outlined_func.getFunctionType().getResults(),
      live_ins.getArrayRef(), launch_op->getAttrs());

  launch_op.replaceAllUsesWith(cluster_func_op);
  launch_op.erase();
}

void ClusterOutliningPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);
  OpBuilder builder(module.getContext());
  module.walk([&](tf_device::ClusterOp cluster) {
    OutlineCluster(cluster, &symbol_table, &builder);
  });
}

void LaunchOutliningPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);
  OpBuilder builder(module.getContext());
  module.walk([&](tf_device::LaunchOp launch) {
    OutlineLaunch(launch, &symbol_table, &builder);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateClusterOutliningPass() {
  return std::make_unique<ClusterOutliningPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateLaunchOutliningPass() {
  return std::make_unique<LaunchOutliningPass>();
}

}  // namespace TFDevice
}  // namespace mlir
