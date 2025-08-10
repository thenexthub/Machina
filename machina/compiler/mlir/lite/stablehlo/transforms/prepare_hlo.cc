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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/fft.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_PREPAREHLOPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

class PrepareHloPass : public impl::PrepareHloPassBase<PrepareHloPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareHloPass);

  void runOnOperation() override;
};

#include "machina/compiler/mlir/lite/stablehlo/transforms/generated_prepare_hlo.inc"
void PrepareHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);

  PopulatePrepareConvPatterns(context, patterns);
  PopulatePrepareReduceWindowPatterns(context, patterns);
  PopulatePrepareSlicePatterns(context, patterns);
  PopulateCustomCallPreparePatterns(context, patterns);
  PopulatePrepareFftPatterns(context, patterns);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareHloPass() {
  return std::make_unique<PrepareHloPass>();
}

static PassRegistration<PrepareHloPass> pass;

}  // namespace odml
}  // namespace mlir
