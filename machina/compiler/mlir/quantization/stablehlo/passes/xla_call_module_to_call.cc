/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_XLACALLMODULETOCALLPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Converts XlaCallModuleOps to func.call.
class XlaCallModuleToCallPass
    : public impl::XlaCallModuleToCallPassBase<XlaCallModuleToCallPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XlaCallModuleToCallPass)

  explicit XlaCallModuleToCallPass() = default;

 private:
  void runOnOperation() override;
};

// Converts XlaCallModuleOps to func.call.
class XlaCallModuleOpToCallOp : public OpRewritePattern<TF::XlaCallModuleOp> {
 public:
  explicit XlaCallModuleOpToCallOp(MLIRContext* context)
      : OpRewritePattern<TF::XlaCallModuleOp>(context) {}

  LogicalResult matchAndRewrite(TF::XlaCallModuleOp op,
                                PatternRewriter& rewriter) const override {
    auto module_op = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module_op);

    auto entry_func_op = dyn_cast_or_null<func::FuncOp>(
        symbol_table.lookup(GetEntryFunctionName(op)));
    if (!entry_func_op) return failure();

    // Replace the XlaCallModuleOp with a new CallOp.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, entry_func_op, op.getArgs());
    return success();
  }
};

void XlaCallModuleToCallPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();
  RewritePatternSet patterns(&getContext());
  patterns.add<XlaCallModuleOpToCallOp>(ctx);
  if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo
