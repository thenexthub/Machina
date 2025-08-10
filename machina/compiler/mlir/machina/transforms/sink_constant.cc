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

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Debug.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"

#define DEBUG_TYPE "tf-executor-sink-constant"

namespace mlir {
namespace TFDevice {

namespace {
using ::mlir::TF::ConstOp;

#define GEN_PASS_DEF_CLUSTERCONSTANTSINKINGPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class ClusterConstantSinkingPass
    : public impl::ClusterConstantSinkingPassBase<ClusterConstantSinkingPass> {
 public:
  explicit ClusterConstantSinkingPass(
      toolchain::function_ref<bool(tf_device::ClusterOp, ElementsAttr)> filter)
      : filter_(filter) {}

  void runOnOperation() override {
    getOperation().walk([filter = filter_](tf_device::ClusterOp cluster) {
      LLVM_DEBUG(toolchain::dbgs() << "Visit " << *cluster.getOperation() << "\n");
      // For each launch op, we find the values used that come from a constant
      // defined above and sink these constants in the region body.
      // The sunk_constant map keeps a mapping from a ConstOp defined above to
      // a sunk clone of it. This allows for reusing a sunk constant with
      // multiple uses in the region.
      toolchain::DenseMap<Value, TF::ConstOp> sunk_constant;
      Region &body = cluster.getBody();
      visitUsedValuesDefinedAbove(body, [&](OpOperand *use) {
        Value constant = use->get();
        auto const_op = dyn_cast_or_null<TF::ConstOp>(constant.getDefiningOp());
        if (!const_op) return;

        // Filter constants using user provided predicate function.
        if (filter && !filter(cluster, const_op.getValue())) return;

        // We found a constant, try to insert it in the map and re-use its
        // cloned value if any.
        auto map_entry = sunk_constant.try_emplace(constant, nullptr);
        if (!map_entry.second) {
          // This constant has already been cloned into the region, reuse it.
          use->set(map_entry.first->getSecond().getResult());
          LLVM_DEBUG(toolchain::dbgs() << "Re-use sunk constant " << use->get()
                                  << "\n     in " << use->get() << "\n");
          if (constant.use_empty()) const_op.erase();
          return;
        }
        if (constant.hasOneUse()) {
          LLVM_DEBUG(toolchain::dbgs() << "Moved constant " << constant << "\n");
          const_op.getOperation()->moveBefore(&body.begin()->front());
          return;
        }
        map_entry.first->getSecond() = const_op.clone();
        body.begin()->getOperations().insert(body.begin()->begin(),
                                             map_entry.first->getSecond());
        use->set(map_entry.first->getSecond().getResult());
        LLVM_DEBUG(toolchain::dbgs() << "Sunk cloned constant " << use->get()
                                << "\n     in " << use->get() << "\n");
      });
    });
  }

 private:
  toolchain::function_ref<bool(tf_device::ClusterOp, ElementsAttr)> filter_;
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateClusterConstantSinkingPass(
    toolchain::function_ref<bool(tf_device::ClusterOp, ElementsAttr)> filter) {
  return std::make_unique<ClusterConstantSinkingPass>(filter);
}

}  // namespace TFDevice
}  // namespace mlir
