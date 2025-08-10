/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/core/transforms/region_to_functional/pass.h"

#include <memory>
#include <utility>

#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/transforms/region_to_functional/impl.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_FUNCTIONALTOREGION
#define GEN_PASS_DEF_REGIONTOFUNCTIONAL
#include "machina/core/transforms/passes.h.inc"

struct RegionToFunctionalPass
    : public impl::RegionToFunctionalBase<RegionToFunctionalPass> {
  explicit RegionToFunctionalPass(bool force_ctl_capture) {
    force_control_capture = force_ctl_capture;
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTable table(getOperation());
    PopulateRegionToFunctionalPatterns(patterns, table, force_control_capture);

    GreedyRewriteConfig config;
    // Use top-down traversal for more efficient conversion. Disable region
    // simplification as all regions are single block.
    config.setUseTopDownTraversal(true);
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    // Iterate until all regions have been outlined. This is guaranteed to
    // terminate because the IR can only hold a finite depth of regions.
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      getOperation()->emitError(getArgument() + " pass failed");
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> CreateRegionToFunctionalPass(bool force_control_capture) {
  return std::make_unique<RegionToFunctionalPass>(force_control_capture);
}

}  // namespace tfg
}  // namespace mlir
