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

#include <memory>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/utils/device_util.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";

#define GEN_PASS_DEF_DEVICEATTRIBUTETOLAUNCHPASS
#include "machina/compiler/mlir/machina/transforms/tf_device_passes.h.inc"

struct DeviceAttributeToLaunch
    : public impl::DeviceAttributeToLaunchPassBase<DeviceAttributeToLaunch> {
  void runOnOperation() override;
};

void WrapOpInLaunch(Operation* op, toolchain::StringRef device) {
  OpBuilder builder(op);

  auto launch_op = builder.create<tf_device::LaunchOp>(
      op->getLoc(), builder.getStringAttr(device),
      /*result_types=*/op->getResultTypes());
  op->replaceAllUsesWith(launch_op);

  launch_op.getBody().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  auto* return_op =
      builder.create<tf_device::ReturnOp>(op->getLoc(), op->getResults())
          .getOperation();
  MLIRContext* context = launch_op.getContext();
  op->removeAttr(StringAttr::get(context, kDeviceAttr));
  op->moveBefore(return_op);
}

void DeviceAttributeToLaunch::runOnOperation() {
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");

  getOperation().walk([&](Operation* op) {
    if (op->getDialect() != tf_dialect) return WalkResult::advance();
    if (auto device = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
      if (!device.getValue().empty()) WrapOpInLaunch(op, device.getValue());
    }
    return WalkResult::advance();
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDeviceAttributeToLaunchPass() {
  return std::make_unique<DeviceAttributeToLaunch>();
}

}  // namespace TFDevice
}  // namespace mlir
