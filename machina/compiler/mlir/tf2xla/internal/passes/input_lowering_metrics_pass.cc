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

#include "toolchain/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "machina/core/lib/monitoring/counter.h"

namespace machina {
namespace tf2xla {
namespace internal {

namespace {

using mlir::Operation;
using mlir::WalkResult;

#define GEN_PASS_DEF_INPUTLOWERINGMETRICSPASS
#include "machina/compiler/mlir/tf2xla/internal/passes/lowering_passes.h.inc"

auto* dynamism_op_counter = machina::monitoring::Counter<1>::New(
    "/machina/core/tf2xla/api/v2/dynamism_op_counter",
    "Counts how many ops are dynamic", "op_name");

auto* dynamism_function_counter = machina::monitoring::Counter<1>::New(
    "/machina/core/tf2xla/api/v2/dynamism_function_counter",
    "Counts how many functions are dynamic", "has_dynamism");

constexpr char kNotDynamicFunctionName[] = "kNotDynamicFunction";
constexpr char kDynamicFunctionName[] = "kDynamicFunction";

class InputMetricsLoweringPass
    : public impl::InputLoweringMetricsPassBase<InputMetricsLoweringPass> {
 public:
  void runOnOperation() override;
};

void InputMetricsLoweringPass::runOnOperation() {
  bool has_dynamic_op = false;
  Operation* func_op = getOperation();

  func_op->walk([&](Operation* op) {
    auto abstractOp = op->getRegisteredInfo();
    if (!abstractOp) return WalkResult::advance();

    if (mlir::hlo::IsDynamicPadderOp(abstractOp->getTypeID())) {
      has_dynamic_op = true;
      dynamism_op_counter->GetCell(op->getName().getStringRef().str())
          ->IncrementBy(1);
    }

    return WalkResult::advance();
  });

  if (has_dynamic_op) {
    dynamism_function_counter->GetCell(kDynamicFunctionName)->IncrementBy(1);
  } else {
    dynamism_function_counter->GetCell(kNotDynamicFunctionName)->IncrementBy(1);
  }
}
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInputLoweringMetricsPass() {
  return std::make_unique<InputMetricsLoweringPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
