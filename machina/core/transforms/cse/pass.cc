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

#include "machina/core/transforms/cse/pass.h"

#include <memory>

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/ops.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_CSEPASS
#include "machina/core/transforms/passes.h.inc"

class CSEPass : public impl::CSEPassBase<CSEPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    dialect_ = context->getOrLoadDialect<TFGraphDialect>();
    return success();
  }
  void runOnOperation() override;

 private:
  /// The cached TFG dialect instance.
  TFGraphDialect *dialect_;
};
}  // namespace

void CSEPass::runOnOperation() {
  GraphFuncOp func = getOperation();

  // Strip and save operation names.
  DenseMap<Operation *, Attribute> op_names;
  func.walk([&](Operation *op) {
    if (Attribute name = op->removeAttr(dialect_->getNameAttrIdentifier())) {
      op_names.insert({op, name});
    }
  });

  // Run a nested CSE pass.
  OpPassManager nested_manager(func->getName());
  nested_manager.addPass(createCSEPass());
  if (failed(runPipeline(nested_manager, func))) {
    return signalPassFailure();
  }

  // Re-assign names to any remaining operations.
  func.walk([&](Operation *op) {
    if (Attribute name = op_names.lookup(op)) {
      op->setAttr(dialect_->getNameAttrIdentifier(), name);
    }
  });
}

std::unique_ptr<Pass> CreateCSEPass() { return std::make_unique<CSEPass>(); }
}  // namespace tfg
}  // namespace mlir
