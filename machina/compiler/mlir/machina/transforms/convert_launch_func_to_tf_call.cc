/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/error_util.h"

namespace mlir {
namespace TFDevice {

namespace {

#define GEN_PASS_DEF_CONVERTLAUNCHFUNCTOTFCALLPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// Rewrites tf_device::LaunchFuncOp into TF::PartitionedCallOp.
struct ConvertLaunchFuncToTFCallPass
    : public impl::ConvertLaunchFuncToTFCallPassBase<
          ConvertLaunchFuncToTFCallPass> {
  void runOnOperation() override;
};

void ConvertLaunchFuncToTFCallPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([&](tf_device::LaunchFuncOp launch) {
    OpBuilder builder(launch);
    auto call_op = builder.create<TF::PartitionedCallOp>(
        module.getLoc(), launch.getResultTypes(), launch.getOperands(),
        /*args_attrs=*/nullptr, /*res_attrs=*/nullptr,
        SymbolRefAttr::get(builder.getContext(), launch.getFunc()),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    call_op->setAttr("device", launch->getAttrOfType<StringAttr>("device"));
    launch.replaceAllUsesWith(call_op);
    launch.erase();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertLaunchFuncToTFCallPass() {
  return std::make_unique<ConvertLaunchFuncToTFCallPass>();
}

}  // namespace TFDevice
}  // namespace mlir
