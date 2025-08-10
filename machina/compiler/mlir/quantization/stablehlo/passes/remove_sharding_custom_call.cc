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
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_REMOVESHARDINGCUSTOMCALLPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

// Include patterns generated from `remove_sharding_custom_call.td`.
#include "machina/compiler/mlir/quantization/stablehlo/passes/remove_sharding_custom_call.inc"

class RemoveShardingCustomCallPass
    : public impl::RemoveShardingCustomCallPassBase<
          RemoveShardingCustomCallPass> {
 public:
  using impl::RemoveShardingCustomCallPassBase<
      RemoveShardingCustomCallPass>::RemoveShardingCustomCallPassBase;

 private:
  void runOnOperation() override;
};

void RemoveShardingCustomCallPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  populateWithGenerated(patterns);

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPatternsGreedily(func_op, frozen_patterns))) {
    func_op.emitWarning() << "Failed to converge "
                          << RemoveShardingCustomCallPass::getArgumentName();
  }
}

}  // namespace mlir::quant::stablehlo
