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

// Legalize TensorFlow Lite StatefulOps to TOSA

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-legalize-tfl-stateful"

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSALEGALIZETFLSTATEFULPASS
#include "machina/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering tfl stateful operators to TOSA
class TosaLegalizeTFLStateful
    : public impl::TosaLegalizeTFLStatefulPassBase<TosaLegalizeTFLStateful> {
 public:
  explicit TosaLegalizeTFLStateful() = default;
  void runOnOperation() override;
};

void TosaLegalizeTFLStateful::runOnOperation() {
  auto moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp.getBodyRegion());

  DenseMap<StringRef, func::FuncOp> symNameToFunction;
  for (auto func : moduleOp.getOps<func::FuncOp>()) {
    symNameToFunction[func.getSymName()] = func;
  }

  toolchain::SmallVector<mlir::TFL::VarHandleOp, 6> handleOps;
  toolchain::SmallVector<mlir::TFL::AssignVariableOp, 6> assignOps;
  toolchain::SmallVector<mlir::TFL::ReadVariableOp, 6> readOps;
  SmallVector<mlir::TFL::CallOnceOp> callOnceOps;
  DenseMap<StringRef, mlir::tosa::VariableOp> symbolRefMap;

  for (auto it : symNameToFunction) {
    auto func = std::get<1>(it);
    // We also want to grab the list of operations to replace.
    for (auto& op : func.getOps()) {
      if (auto handle = dyn_cast<mlir::TFL::VarHandleOp>(op))
        handleOps.push_back(handle);
      if (auto assign = dyn_cast<mlir::TFL::AssignVariableOp>(op))
        assignOps.push_back(assign);
      if (auto read = dyn_cast<mlir::TFL::ReadVariableOp>(op))
        readOps.push_back(read);
    }
  }

  for (auto func : moduleOp.getOps<func::FuncOp>()) {
    for (auto init : func.getOps<mlir::TFL::CallOnceOp>()) {
      callOnceOps.push_back(init);
    }
  }

  // Look through the initialization functions and find the assigned values
  // for each handle, save out the constant value.
  for (auto init : callOnceOps) {
    auto findInitFunc =
        symNameToFunction.find(init.getSessionInitFunctionAttr());
    if (findInitFunc == symNameToFunction.end()) {
      init.emitError("unable to find initialization function: ");
      continue;
    }
    func::FuncOp initFunc = std::get<1>(*findInitFunc);
    for (auto assign : initFunc.getOps<mlir::TFL::AssignVariableOp>()) {
      // 1. var_handle part
      auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
          assign.getResourceId().getDefiningOp());
      if (!handle) continue;

      // 2. pseudo_const part
      DenseElementsAttr constant;
      if (!matchPattern(assign.getValue(), m_Constant(&constant))) {
        // Quantized types we can not use the m_Constant matcher.
        if (auto constOp = dyn_cast<mlir::TFL::QConstOp>(
                assign.getValue().getDefiningOp())) {
          constant = cast<DenseElementsAttr>(constOp.getValue());
        }
      }
      if (!constant) continue;

      // Create TOSA VariableOps
      auto name = handle.getSharedName();
      auto global = builder.create<mlir::tosa::VariableOp>(
          handle.getLoc(), name, constant.getType(), constant);
      symbolRefMap[name] = global;
    }
  }
  // TF::CallOnceOps are no longer needed as we have already extracted their
  // state.
  for (auto op : callOnceOps) op.erase();

  // Replace the assign ops with a tosa store operation.
  for (auto assign : assignOps) {
    auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
        assign.getResourceId().getDefiningOp());
    if (!handle) continue;

    Value value = assign.getValue();
    auto globalOpIt = symbolRefMap.find(handle.getSharedName());
    if (globalOpIt == symbolRefMap.end()) {
      assign->emitError(
          "unable to find corresponding TosaOp for op's VarHandle");
      continue;
    }
    auto globalOp = std::get<1>(*globalOpIt);

    builder.setInsertionPoint(assign);
    auto variableType = mlir::tosa::getVariableType(globalOp);
    if (variableType != value.getType()) {
      value = builder
                  .create<UnrealizedConversionCastOp>(assign.getLoc(),
                                                      variableType, value)
                  .getResult(0);
    }

    builder.create<mlir::tosa::VariableWriteOp>(
        assign.getLoc(), toolchain::StringRef(globalOp.getName()), value);
    assign.erase();
  }

  for (auto read : readOps) {
    auto handle =
        dyn_cast<mlir::TFL::VarHandleOp>(read.getResourceId().getDefiningOp());
    if (!handle) continue;

    auto globalOpIt = symbolRefMap.find(handle.getSharedName());
    if (globalOpIt == symbolRefMap.end()) continue;
    auto globalOp = std::get<1>(*globalOpIt);

    builder.setInsertionPoint(read);

    Value load = builder.create<mlir::tosa::VariableReadOp>(
        read.getLoc(), mlir::tosa::getVariableType(globalOp),
        toolchain::StringRef(globalOp.getName()));

    if (read.getType() != load.getType()) {
      load = builder
                 .create<UnrealizedConversionCastOp>(read.getLoc(),
                                                     read.getType(), load)
                 .getResult(0);
    }
    read.getResult().replaceAllUsesWith(load);
    read.erase();
  }

  for (auto handle : handleOps) {
    if (handle.getResult().use_empty()) {
      handle.erase();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTFLStateful pass.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFLStatefulPass() {
  return std::make_unique<TosaLegalizeTFLStateful>();
}

}  // namespace tosa
}  // namespace mlir
