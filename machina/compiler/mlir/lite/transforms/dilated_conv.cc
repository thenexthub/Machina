/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/compiler/mlir/lite/transforms/dilated_conv.h"

#include <utility>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_IDENTIFYDILATEDCONVPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

struct IdentifyDilatedConvPass
    : public impl::IdentifyDilatedConvPassBase<IdentifyDilatedConvPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentifyDilatedConvPass)
  void runOnOperation() override;
};

void IdentifyDilatedConvPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertTFDilatedConvOp<TF::Conv2DOp>,
               ConvertTFDilatedConvOp<TF::DepthwiseConv2dNativeOp>>(
      &getContext());
  (void)applyPatternsGreedily(func, std::move(patterns));
}
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> CreateIdentifyDilatedConvPass() {
  return std::make_unique<IdentifyDilatedConvPass>();
}

}  // namespace TFL
}  // namespace mlir
