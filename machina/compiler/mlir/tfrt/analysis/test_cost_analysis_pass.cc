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
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/analysis/cost_analysis.h"

namespace machina {
namespace tfrt_compiler {

class TestCostAnalysis
    : public mlir::PassWrapper<TestCostAnalysis,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  toolchain::StringRef getArgument() const final {
    return "tfrt-test-cost-analysis";
  }
  toolchain::StringRef getDescription() const final {
    return "Add remarks based on cost analysis for testing purpose.";
  }
  void runOnOperation() override {
    const auto& cost_analysis = getAnalysis<CostAnalysis>();

    auto func_op = getOperation();
    for (auto& op : func_op.front()) {
      op.emitRemark() << "Cost: " << cost_analysis.GetCost(&op);
    }
  }
};

static mlir::PassRegistration<TestCostAnalysis> pass;

}  // namespace tfrt_compiler
}  // namespace machina
