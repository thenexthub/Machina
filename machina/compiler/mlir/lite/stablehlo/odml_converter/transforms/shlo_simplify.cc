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

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/stablehlo/odml_converter/folders.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_SHLOSIMPLIFYPASS
#include "machina/compiler/mlir/lite/stablehlo/odml_converter/passes.h.inc"
#include "machina/compiler/mlir/lite/stablehlo/odml_converter/transforms/generated_shlo_simplify.inc"

// Performs misc odml "cleanup" on shlo dialect. This is a functional standin
// for canonicalization and folding which is not offered directly by the
// shlo implementation.
class SHLOSimplifyPass : public impl::SHLOSimplifyPassBase<SHLOSimplifyPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SHLOSimplifyPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    PopulateFolderPatterns(patterns);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateSHLOSimplifyPass() {
  return std::make_unique<SHLOSimplifyPass>();
}

}  // namespace odml
}  // namespace mlir
