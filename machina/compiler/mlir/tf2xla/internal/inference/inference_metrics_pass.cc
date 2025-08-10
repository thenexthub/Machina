/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/lib/monitoring/counter.h"

namespace mlir {
namespace tf2xla {
namespace internal {

auto* has_tpu_partitioned_call_streamz =
    machina::monitoring::Counter<1>::New(
        "/machina/core/tf2xla/internal/inference/tpu_partitioned_call",
        "Whether the model has TPUPartitionedCallOp.",
        "has_tpu_partitioned_call");

namespace {

#define GEN_PASS_DEF_INFERENCEMETRICSPASS
#include "machina/compiler/mlir/tf2xla/internal/inference/inference_passes.h.inc"

class InferenceMetricsPass
    : public impl::InferenceMetricsPassBase<InferenceMetricsPass> {
 public:
  void runOnOperation() override;
};

void InferenceMetricsPass::runOnOperation() {
  bool has_tpu_partitioned_call = false;
  ModuleOp module = getOperation();

  for (auto func_op : module.getOps<func::FuncOp>()) {
    func_op->walk(
        [&](TF::TPUPartitionedCallOp op) { has_tpu_partitioned_call = true; });

    if (has_tpu_partitioned_call) break;
  }

  std::string has_tpu_partitioned_call_str =
      has_tpu_partitioned_call ? "true" : "false";
  has_tpu_partitioned_call_streamz->GetCell(has_tpu_partitioned_call_str)
      ->IncrementBy(1);
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateInferenceMetricsPass() {
  return std::make_unique<InferenceMetricsPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace mlir
