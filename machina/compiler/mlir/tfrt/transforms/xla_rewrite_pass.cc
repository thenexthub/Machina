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

#include <cstdint>
#include <memory>
#include <utility>

#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "machina/core/ir/types/dialect.h"

namespace machina {
namespace tfrt_compiler {
namespace {

struct RewriteStatefulPartitionedCallToXlaLaunchOnCpu
    : public mlir::OpRewritePattern<mlir::TF::StatefulPartitionedCallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::StatefulPartitionedCallOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (auto xla_must_compile =
            op->getAttrOfType<mlir::BoolAttr>("_XlaMustCompile");
        !xla_must_compile || !xla_must_compile.getValue()) {
      return mlir::failure();
    }

    toolchain::StringRef device = GetDeviceOrEmpty(op);

    if (device.empty() || !device.contains("CPU")) return mlir::failure();

    toolchain::SmallVector<int64_t> constants;
    toolchain::SmallVector<int64_t> resources;

    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto value = op.getOperand(i);
      if (toolchain::isa<mlir::tf_type::ResourceType>(
              toolchain::cast<mlir::TensorType>(value.getType()).getElementType())) {
        resources.push_back(i);
      } else if (auto* def = value.getDefiningOp();
                 def && toolchain::isa<mlir::TF::ConstOp>(def)) {
        constants.push_back(i);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::TF::XlaLaunchV2Op>(
        op, op.getResultTypes(), op.getOperands(),
        rewriter.getI64ArrayAttr(constants),
        rewriter.getI64ArrayAttr(resources), op.getFAttr());

    return mlir::success();
  }
};

struct TfrtXlaRewritePass
    : public mlir::PassWrapper<TfrtXlaRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  toolchain::StringRef getArgument() const override { return "tfrt-xla-rewrite"; }

  toolchain::StringRef getDescription() const override {
    return "rewrites for XLA host ops.";
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TfrtXlaRewritePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<RewriteStatefulPartitionedCallToXlaLaunchOnCpu>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfrtXlaRewritePass() {
  return std::make_unique<TfrtXlaRewritePass>();
}

static mlir::PassRegistration<TfrtXlaRewritePass> register_pass(
    CreateTfrtXlaRewritePass);

}  // namespace tfrt_compiler
}  // namespace machina
