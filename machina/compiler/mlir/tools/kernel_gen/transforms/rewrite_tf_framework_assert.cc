/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

// Converts tf_framework.assert to a conditional branch that reports an error to
// OpKernelContext and creates a fake memref using NullMemRefOp.
class TFAssertOpConverter : public OpConversionPattern<TFAssertOp> {
 public:
  using OpConversionPattern<TFAssertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TFAssertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Split the block to insert CondBr.
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    Block *split_block = rewriter.splitBlock(
        rewriter.getInsertionBlock(), std::next(rewriter.getInsertionPoint()));

    auto func = op->getParentOfType<func::FuncOp>();
    Block *error_reporting_block =
        rewriter.createBlock(&func.getRegion(), {}, {});
    rewriter.create<ReportErrorOp>(loc, adaptor.getCtx(),
                                   adaptor.getErrorCode(), adaptor.getMsg());

    SmallVector<Value, 2> null_memrefs;
    for (auto type : func.getFunctionType().getResults()) {
      null_memrefs.push_back(rewriter.create<NullMemRefOp>(loc, type));
    }
    rewriter.create<func::ReturnOp>(loc, null_memrefs);

    rewriter.restoreInsertionPoint(ip);
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getArg(), split_block, std::nullopt, error_reporting_block,
        std::nullopt);
    return success();
  }
};

#define GEN_PASS_DEF_REWRITETFFRAMEWORKASSERT
#include "machina/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

bool IsNotInsideTfEntryFunction(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return !func->hasAttrOfType<UnitAttr>(TFFrameworkDialect::kTFEntryAttrName);
}
// All contained `tf_framework.assert` operations are rewritten into calls to
// `tf_framework.report_error` and the required control flow to make
// execution of the function terminate.
class RewriteTFFrameworkAssertPass
    : public impl::RewriteTFFrameworkAssertBase<RewriteTFFrameworkAssertPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate patterns.
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TFAssertOpConverter>(context);
    PopulateEmbedTFFrameworkAssertPattern(&patterns);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<tf_framework::TFFrameworkDialect, func::FuncDialect,
                           cf::ControlFlowDialect>();
    target.addIllegalOp<TFAssertOp>();
    target.addDynamicallyLegalOp<cf::AssertOp>(IsNotInsideTfEntryFunction);

    if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateRewriteTFFrameworkAssert() {
  return std::make_unique<RewriteTFFrameworkAssertPass>();
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
