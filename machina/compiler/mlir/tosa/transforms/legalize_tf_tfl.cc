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

// Legalize TensorFlow and TensorFlow Lite to TOSA

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/tosa/transforms/legalize_common.h"
#include "machina/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "machina/compiler/mlir/tosa/transforms/passes.h"

namespace mlir {
namespace tosa {

namespace {

#define GEN_PASS_DEF_TOSALEGALIZETFTFLPASS
#include "machina/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering to TOSA dialect
class LegalizeTFTFL : public impl::TosaLegalizeTFTFLPassBase<LegalizeTFTFL> {
 public:
  explicit LegalizeTFTFL() = default;
  void runOnOperation() override;
};

void LegalizeTFTFL::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateLegalizeTFPatterns(ctx, patterns);
  populateLegalizeTFLPatterns(ctx, patterns);

  func::FuncOp func = getOperation();
  if (ApplyPatternsWithShapeResolution(func, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

}  // anonymous namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTFL pass.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFTFLPass() {
  return std::make_unique<LegalizeTFTFL>();
}

}  // namespace tosa
}  // namespace mlir
