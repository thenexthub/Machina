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
#include "machina/compiler/mlir/tfrt/transforms/set_shape_invariant_in_while_ops.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace machina {
namespace tfrt_compiler {
namespace {

class SetShapeInvariantInWhileOps
    : public mlir::PassWrapper<SetShapeInvariantInWhileOps,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetShapeInvariantInWhileOps)

  void runOnOperation() override {
    mlir::func::FuncOp func_op = getOperation();

    auto shape_invariant = mlir::UnitAttr::get(&getContext());

    func_op.walk([&](mlir::TF::WhileOp op) {
      // Skip tf.While op on TPU.
      if (!op->hasAttr("_tpu_replicate")) {
        op.setShapeInvariantAttr(shape_invariant);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateSetShapeInvariantInWhileOps() {
  return std::make_unique<SetShapeInvariantInWhileOps>();
}

}  // namespace tfrt_compiler
}  // namespace machina
