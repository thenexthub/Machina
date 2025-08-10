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
#include "machina/compiler/mlir/tfrt/transforms/mlrt/assign_op_key.h"

#include <stdint.h>

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/constants.h"
#include "machina/compiler/mlir/tfrt/transforms/mlrt/util.h"

namespace machina {
namespace mlrt_compiler {
namespace {

class AssignOpKeyPass
    : public mlir::PassWrapper<AssignOpKeyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  AssignOpKeyPass() = default;
  AssignOpKeyPass& operator=(const AssignOpKeyPass&) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignOpKeyPass)

 private:
  toolchain::StringRef getArgument() const final { return "tf-mlrt-assign-op-key"; }
  toolchain::StringRef getDescription() const final {
    return "tf-mlrt-assign-op-key";
  }

  void runOnOperation() override;
};

void AssignOpKeyPass::runOnOperation() {
  auto module = getOperation();
  mlir::OpBuilder builder(module);

  int32_t op_key = 0;
  module.walk([&builder, &op_key](mlir::Operation* op) mutable {
    if (UseFallback(op)) {
      op->setAttr(machina::tfrt_compiler::kOpKeyAttrName,
                  builder.getI32IntegerAttr(op_key));
      op_key++;
    }
  });
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAssignOpKeyPass() {
  return std::make_unique<AssignOpKeyPass>();
}

}  // namespace mlrt_compiler
}  // namespace machina
