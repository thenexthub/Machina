/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

// Converts TF While to TFL While with single call in body and cond.

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEWHILEPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

// Legalize TF While to TFL While with calls to the original functions from the
// cond and body regions.
struct LegalizeWhilePass
    : public impl::LegalizeWhilePassBase<LegalizeWhilePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeWhilePass)
  void RunOnFunction(func::FuncOp func);

  void runOnOperation() override {
    for (auto op : getOperation().getOps<func::FuncOp>()) RunOnFunction(op);
  }
};

}  // namespace

// Inserts call to the given function into the 'region'.
void CreateRegionWithCall(func::FuncOp func, Region& region, Location loc) {
  OpBuilder builder(region);
  auto block = builder.createBlock(&region);
  SmallVector<Value, 4> new_operands;
  for (Type t : func.getFunctionType().getInputs())
    new_operands.push_back(block->addArgument(t, loc));
  auto call = func::CallOp::create(builder, loc, func, new_operands);
  YieldOp::create(builder, loc, call.getResults());
  // Mark old function as private so that it can be DCE'd if not called.
  func.setPrivate();
}

void RunOnWhile(TF::WhileOp while_op) {
  Operation* op = while_op.getOperation();
  // Create new TFL While op that will be used to replace TF While op.
  OpBuilder builder(op);
  auto new_op =
      TFL::WhileOp::create(builder, op->getLoc(), op->getResultTypes(),
                           op->getOperands(), while_op.getIsStateless());
  Location loc = while_op->getLoc();
  CreateRegionWithCall(while_op.cond_function(), new_op.getCond(), loc);
  CreateRegionWithCall(while_op.body_function(), new_op.getBody(), loc);

  op->replaceAllUsesWith(new_op.getResults());
  op->erase();
}

void LegalizeWhilePass::RunOnFunction(func::FuncOp func) {
  // Convert all TF WhileOps inside the function body to TFL While ops.
  func.getBody().walk([](TF::WhileOp while_op) { RunOnWhile(while_op); });
}

// Creates an instance of the TensorFlow While to TFLite While pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFWhilePass() {
  return std::make_unique<LegalizeWhilePass>();
}

static PassRegistration<LegalizeWhilePass> pass;

}  // namespace TFL
}  // namespace mlir
