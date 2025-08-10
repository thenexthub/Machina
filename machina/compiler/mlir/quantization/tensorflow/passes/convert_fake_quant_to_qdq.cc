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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain  // IWYU pragma: keep, for applyPatternsGreedily
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/quantization/machina/utils/fake_quant_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"

namespace mlir {
namespace quant {
namespace {

class ConvertFakeQuantToQdqPass
    : public PassWrapper<ConvertFakeQuantToQdqPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFakeQuantToQdqPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-convert-fake-quant-to-qdq";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert Fake Quant op to quant.qcast and quant.dcast pairs";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<quant::QuantDialect>();
    registry.insert<mlir::quant::ir::TFQuantDialect>();
  }

  void runOnOperation() override;
};

static PassRegistration<ConvertFakeQuantToQdqPass> pass;

void ConvertFakeQuantToQdqPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  func::FuncOp func = getOperation();

  if (failed(
          ConvertFakeQuantOps(func, ctx, /*use_fake_quant_num_bits=*/false))) {
    func.emitError() << "quant-convert-fake-quant-to-qdq pass failed.";
    signalPassFailure();
  }

  // For removing dead FakeQuant* ops
  RewritePatternSet patterns(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertFakeQuantToQdqPass() {
  return std::make_unique<ConvertFakeQuantToQdqPass>();
}

}  // namespace quant
}  // namespace mlir
