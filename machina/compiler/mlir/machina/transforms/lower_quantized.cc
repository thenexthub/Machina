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

// Rewrites ops that require quantized inputs or outputs to ops that allow
// non-quantized inputs and outputs.

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/lower_tf.h"

#define DEBUG_TYPE "tf-lower-quantized"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_LOWERQUANTIZEDPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class LowerQuantizedPass
    : public impl::LowerQuantizedPassBase<LowerQuantizedPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::TF::PopulateLoweringQuantizedPatterns(&getContext(), &patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLowerQuantizedPass() {
  return std::make_unique<LowerQuantizedPass>();
}

}  // namespace TF
}  // namespace mlir
