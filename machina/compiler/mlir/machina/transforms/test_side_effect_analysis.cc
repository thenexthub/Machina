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

#include <cstdint>
#include <memory>
#include <string>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringExtras.h"
#include "toolchain/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/side_effect_analysis.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"

namespace mlir {
namespace tf_test {

namespace {

struct TestSideEffectAnalysisPass
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TestSideEffectAnalysisPass, TF::SideEffectAnalysis> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSideEffectAnalysisPass)

  void runOnFunction(func::FuncOp func,
                     const TF::SideEffectAnalysis::Info& analysis) {
    int64_t next_id = 0;
    toolchain::SmallDenseMap<Operation*, int64_t, 8> ids;
    func.walk([&](Operation* op) {
      ids[op] = next_id++;
      op->emitRemark("ID: ") << ids[op];
    });
    auto join_ids = [&](const toolchain::ArrayRef<Operation*> ops) {
      toolchain::SmallVector<std::string, 8> id_vec;
      id_vec.reserve(ops.size());
      for (auto op : ops) id_vec.push_back(std::to_string(ids[op]));
      return toolchain::join(id_vec, ",");
    };
    func.walk([&](Operation* op) {
      if (!analysis.DirectControlPredecessors(op).empty()) {
        op->emitRemark("Predecessors: ")
            << "{" << join_ids(analysis.DirectControlPredecessors(op)) << "}";
      }
      if (!analysis.DirectControlSuccessors(op).empty()) {
        op->emitRemark("Successors: ")
            << "{" << join_ids(analysis.DirectControlSuccessors(op)) << "}";
      }
      if (toolchain::isa<func::ReturnOp>(op)) {
        op->emitRemark("Sinks: ")
            << "{" << join_ids(analysis.ControlSinks()) << "}";
      }
    });
  }

  StringRef getArgument() const final { return "tf-test-side-effect-analysis"; }
  StringRef getDescription() const final {
    return "Test pass for analyzing side-effect analysis result";
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTestSideEffectAnalysisPass() {
  return std::make_unique<TestSideEffectAnalysisPass>();
}

}  // namespace tf_test
}  // namespace mlir
