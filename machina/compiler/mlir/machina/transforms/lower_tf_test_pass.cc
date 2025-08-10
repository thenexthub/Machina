/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/lower_tf.h"
#include "machina/compiler/mlir/machina/transforms/test_passes.h"

namespace mlir {
namespace tf_test {
namespace {

#define GEN_PASS_DEF_TESTMACHINALOWERTFPASS
#include "machina/compiler/mlir/machina/transforms/test_passes.h.inc"

// Lowers some of the TensorFlow operations that can be represented using other
// TensorFlow operations.
struct LowerTF : public impl::TestTensorFlowLowerTFPassBase<LowerTF> {
  void runOnOperation() override {
    // Add lowering patterns to the list.
    RewritePatternSet patterns(&getContext());
    if (default_patterns_) {
      mlir::TF::PopulateLoweringTFPatterns(&getContext(), &patterns);
    }
    if (pre_hlo_patterns_) {
      mlir::TF::PopulateTFLoweringBeforeHLOPatterns(&getContext(), &patterns);
    }

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTestTFLowerTFPass() {
  return std::make_unique<LowerTF>();
}

}  // namespace tf_test
}  // namespace mlir
