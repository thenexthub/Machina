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

#include <memory>

#include "mlir/Analysis/AliasAnalysis.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/IR/SCF.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/Transforms/Transforms.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_FUSEINNERPARALLELLOOPSPASS
#include "machina/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct FuseInnerParallelLoopsPass
    : impl::FuseInnerParallelLoopsPassBase<FuseInnerParallelLoopsPass> {
  void runOnOperation() override {
    auto &alias_analysis = getAnalysis<AliasAnalysis>();
    auto may_alias = [&](Value val1, Value val2) -> bool {
      return !alias_analysis.alias(val1, val2).isNo();
    };
    getOperation().walk([&](mlir::scf::ParallelOp op) {
      mlir::scf::naivelyFuseParallelOps(op.getRegion(), may_alias);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseInnerParallelLoopsPass() {
  return std::make_unique<FuseInnerParallelLoopsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
