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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gelu.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_LEGALIZECHLOTOTFLPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

class LegalizeChloToTflPass
    : public impl::LegalizeChloToTflPassBase<LegalizeChloToTflPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeChloToTflPass)

  void runOnOperation() override {
    auto func = getOperation();
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerGELU>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeChloToTflPass() {
  return std::make_unique<LegalizeChloToTflPass>();
}

static PassRegistration<LegalizeChloToTflPass> pass;

}  // namespace odml
}  // namespace mlir
