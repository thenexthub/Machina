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

#include <utility>

#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "machina/xla/mlir_hlo/mhlo/transforms/rewriters.h"

//===----------------------------------------------------------------------===//
// The unfuse-mhlo-batch-norm Pass.
//===----------------------------------------------------------------------===//

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_UNFUSEMHLOBATCHNORMPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

class UnfuseMhloBatchNormPass
    : public impl::UnfuseMhloBatchNormPassBase<UnfuseMhloBatchNormPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnfuseMhloBatchNormPass)

  explicit UnfuseMhloBatchNormPass() = default;

 private:
  void runOnOperation() override;
};

void UnfuseMhloBatchNormPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  mhlo::populateUnfuseBatchNormPatterns(ctx, &patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
}  // namespace

}  // namespace mlir::quant::stablehlo
