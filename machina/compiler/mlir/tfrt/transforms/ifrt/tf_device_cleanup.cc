/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"

namespace machina {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFDEVICECLEANUPPASS
#define GEN_PASS_DECL_TFDEVICECLEANUPPASS
#include "machina/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class TfDeviceCleanupPass
    : public impl::TfDeviceCleanupPassBase<TfDeviceCleanupPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    func.walk([](mlir::Operation* op) {
      if (toolchain::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) {
        op->removeAttr("device");
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfDeviceCleanupPass() {
  return std::make_unique<TfDeviceCleanupPass>();
}

}  // namespace ifrt_serving
}  // namespace machina
