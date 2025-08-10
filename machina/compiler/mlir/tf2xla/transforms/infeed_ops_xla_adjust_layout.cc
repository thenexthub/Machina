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

#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/set_tpu_infeed_layout.h"
#include "machina/xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "machina/xla/layout.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "machina/xla/shape.h"
#include "machina/xla/stream_executor/tpu/c_api_conversions.h"
#include "machina/xla/stream_executor/tpu/tpu_api.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_INFEEDOPSXLAADJUSTLAYOUT
#include "machina/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

class InfeedsOpsXlaAdjustLayout
    : public impl::InfeedOpsXlaAdjustLayoutBase<InfeedsOpsXlaAdjustLayout> {
 public:
  void runOnOperation() override;

 private:
  static void runOnInfeedOp(::mlir::mhlo::InfeedOp op) {
    OpBuilder builder(op.getContext());
    SmallVector<Type> result_types(op.getResultTypes().begin(),
                                   op.getResultTypes().end());
    if (!op->getAttr("layout")) {
      auto layout = mlir::GetTPUInfeedLayout(result_types, builder);
      if (failed(layout)) return;

      op->setAttr("layout", layout.value());
    }
  }
};

void InfeedsOpsXlaAdjustLayout::runOnOperation() {
  getOperation().walk(runOnInfeedOp);
}

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateInfeedsOpsXlaAdjustLayoutPass() {
  return std::make_unique<InfeedsOpsXlaAdjustLayout>();
}

}  // namespace mhlo
}  // namespace mlir
