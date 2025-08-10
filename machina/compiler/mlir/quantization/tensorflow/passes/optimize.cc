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
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"  // IWYU pragma: keep - required to use `IsSplatValueEqual`.
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"

namespace mlir::quant {
namespace {

// Applies optimization after quantization.
class OptimizePass
    : public PassWrapper<OptimizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-optimize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Applies optimization after quantization";
  }

  void runOnOperation() override;
};

#include "machina/compiler/mlir/quantization/machina/passes/optimize.inc"

void OptimizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  auto func = getOperation();
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizePass() {
  return std::make_unique<OptimizePass>();
}

static PassRegistration<OptimizePass> pass;

}  // namespace mlir::quant
