/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include <optional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tf2xla/transforms/passes.h"
#include "machina/compiler/tf2xla/kernels/rng_converter_utils.h"
#include "machina/xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

namespace {

#define GEN_PASS_DEF_TFXLADEVICESPECIFICTRANSFORMS
#include "machina/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

class TFXLADeviceSpecificTransforms
    : public impl::TFXLADeviceSpecificTransformsBase<
          TFXLADeviceSpecificTransforms> {
 public:
  explicit TFXLADeviceSpecificTransforms(std::optional<StringRef> device_type) {
    if (device_type.has_value()) {
      device_type_ = device_type.value().str();
    }
  }
  void runOnOperation() override;

 private:
  LogicalResult ConvertGetAlgOp(TF::StatelessRandomGetAlgOp get_alg_op);
};

LogicalResult TFXLADeviceSpecificTransforms::ConvertGetAlgOp(
    TF::StatelessRandomGetAlgOp get_alg_op) {
  if (!device_type_.hasValue()) return failure();

  xla::RandomAlgorithm xla_rng =
      machina::DefaultRngAlgForDeviceType(device_type_);
  machina::Algorithm machina_rng =
      machina::ToTensorflowAlgorithm(xla_rng);

  OpBuilder opbuilder(get_alg_op);

  auto tf_const = opbuilder.create<TF::ConstOp>(
      get_alg_op->getLoc(), opbuilder.getI32IntegerAttr((int)machina_rng));

  get_alg_op->replaceAllUsesWith(tf_const);
  get_alg_op->erase();
  return success();
}

void TFXLADeviceSpecificTransforms::runOnOperation() {
  if (!device_type_.hasValue()) return;
  auto func_op = getOperation();

  auto walk_result = func_op->walk([&](TF::StatelessRandomGetAlgOp op) {
    if (failed(ConvertGetAlgOp(op))) {
      op->emitOpError(
          "Could not convert and remove Device specific information");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTFXLADeviceSpecificTransformsPass(std::optional<StringRef> device_type) {
  return std::make_unique<TFXLADeviceSpecificTransforms>(device_type);
}

}  // namespace mhlo
}  // namespace mlir
