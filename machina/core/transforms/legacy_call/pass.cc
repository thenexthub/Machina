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

#include "machina/core/transforms/legacy_call/pass.h"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/core/ir/interfaces.h"
#include "machina/core/ir/ops.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_LIFTLEGACYCALL
#include "machina/core/transforms/passes.h.inc"

class LiftLegacyCallPass : public impl::LiftLegacyCallBase<LiftLegacyCallPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    tfg_legacy_call_id_ = StringAttr::get(context, "tfg.legacy_call");
    return success();
  }

  void runOnOperation() override {
    FunctionTable table(getOperation());
    for (Operation &op : getOperation().getOps()) {
      op.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IntrinsicOperation>() ||
            !table.IsLegacyCall(op))
          return;

        op->setAttr(tfg_legacy_call_id_,
                    FlatSymbolRefAttr::get(&getContext(),
                                           op->getName().stripDialect()));
      });
    }
  }

 private:
  // The cached identifier of the legacy call tag.
  StringAttr tfg_legacy_call_id_;
};
}  // namespace
std::unique_ptr<Pass> CreateLiftLegacyCallPass() {
  return std::make_unique<LiftLegacyCallPass>();
}
}  // namespace tfg
}  // namespace mlir
