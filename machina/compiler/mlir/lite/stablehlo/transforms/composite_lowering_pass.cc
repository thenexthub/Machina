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

#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/composite_avg_pool.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

namespace {

// This file is generated from `passes.td` and provides the implementation base
// class.
#define GEN_PASS_DEF_COMPOSITELOWERINGPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

class CompositeLoweringPass
    : public impl::CompositeLoweringPassBase<CompositeLoweringPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositeLoweringPass);

  void runOnOperation() override;
};

#include "machina/compiler/mlir/lite/stablehlo/transforms/generated_composite_lowering.inc"

void CompositeLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  populateWithGenerated(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("Composite lowering pass failed.");
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateCompositeLoweringPass() {
  return std::make_unique<CompositeLoweringPass>();
}

// Registers the pass implementation
static PassRegistration<CompositeLoweringPass> pass;

}  // namespace odml
}  // namespace mlir
