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

#include "machina/compiler/mlir/lite/experimental/tac/transforms/device_transform_gpu.h"

#include <memory>
#include <utility>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "machina/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

struct DeviceTransformGPUPass
    : public mlir::PassWrapper<DeviceTransformGPUPass,
                               OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceTransformGPUPass)

  toolchain::StringRef getArgument() const final {
    return "tfl-device-transform-gpu";
  }
  toolchain::StringRef getDescription() const final {
    return "Suitable transformation for gpu only.";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }
  void runOnOperation() override;
};

void DeviceTransformGPUPass::runOnOperation() {
  auto func = getOperation();
  auto* ctx = &getContext();
  RewritePatternSet patterns = GetHardwareRewritePatternsGPU(ctx);
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

RewritePatternSet GetHardwareRewritePatternsGPU(MLIRContext* context) {
  GpuHardware gpu_hardware;
  return gpu_hardware.GetTransformations(context);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDeviceTransformGPUPass() {
  return std::make_unique<DeviceTransformGPUPass>();
}

static PassRegistration<DeviceTransformGPUPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
