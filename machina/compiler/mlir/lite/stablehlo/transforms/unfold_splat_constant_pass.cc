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

#include <cstdint>
#include <memory>

#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

#define DEBUG_TYPE "unfold-splat-constant-pass"

#define GEN_PASS_DEF_UNFOLDSPLATCONSTANTPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

// Undo the MHLO::BroadcastInDimOp folding pattern on splat tensor.
// TODO(b/295966255): Remove this pass after moving MHLO folders to a separate
// pass and folders are not applied by default.
class UnfoldSplatConstantPass
    : public impl::UnfoldSplatConstantPassBase<UnfoldSplatConstantPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();

    mlir::OpBuilder op_builder(&module.getBodyRegion());
    // Cannot use the pattern rewriter because the driver applies folders by
    // default.
    module.walk([&](mhlo::ConstantOp const_op) {
      UnfoldSplatConstant(&op_builder, const_op);
    });
  }

 private:
  void UnfoldSplatConstant(mlir::OpBuilder* op_builder,
                           mhlo::ConstantOp const_op) const {
    auto splat_elements_attr =
        mlir::dyn_cast<SplatElementsAttr>(const_op.getValue());
    if (!splat_elements_attr) {
      return;
    }
    if (splat_elements_attr.getNumElements() == 1) {
      return;
    }
    auto element_type = splat_elements_attr.getType().getElementType();
    if (mlir::isa<ComplexType>(element_type) ||
        mlir::isa<quant::QuantizedType>(element_type)) {
      return;
    }
    op_builder->setInsertionPoint(const_op);
    Value scalar = op_builder->create<mhlo::ConstantOp>(
        const_op->getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get(/*shape=*/{}, element_type),
            splat_elements_attr.getSplatValue<Attribute>()));
    auto broadcast_dims = DenseIntElementsAttr::get(
        RankedTensorType::get(/*shape=*/{0}, op_builder->getI64Type()),
        toolchain::SmallVector<int64_t>{});
    mhlo::BroadcastInDimOp broadcast_in_dim_op =
        op_builder->create<mhlo::BroadcastInDimOp>(
            const_op->getLoc(), splat_elements_attr.getType(), scalar,
            broadcast_dims);
    const_op->replaceAllUsesWith(broadcast_in_dim_op);
    const_op->erase();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldSplatConstantPass() {
  return std::make_unique<UnfoldSplatConstantPass>();
}

static PassRegistration<UnfoldSplatConstantPass> pass;

}  // namespace odml
}  // namespace mlir
