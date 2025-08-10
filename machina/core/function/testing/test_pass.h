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

#ifndef MACHINA_CORE_FUNCTION_TESTING_TEST_PASS_H_
#define MACHINA_CORE_FUNCTION_TESTING_TEST_PASS_H_

#include <memory>

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/ops.h"
#include "machina/core/ir/tf_op_wrapper.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace core {
namespace function {
namespace testing {

// A simple testing pass for BinaryFunction that replaces an AddV2 node named
// `x_plus_y` with a Mul one.
struct TestPassTfgDialect
    : public mlir::PassWrapper<TestPassTfgDialect,
                               mlir::OperationPass<mlir::ModuleOp>> {
  TestPassTfgDialect() = default;

  toolchain::StringRef getArgument() const final { return "test-pass"; }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder builder(module);
    mlir::tfg::TFGraphDialect* dialect =
        builder.getContext()->getOrLoadDialect<mlir::tfg::TFGraphDialect>();

    mlir::Operation* target = nullptr;
    module->walk([&target](mlir::tfg::TFOp op) {
      if (op.nameAttr() == nullptr) {
        return;
      }
      if (op.name() != "x_plus_y") {
        return;
      }
      target = op.getOperation();
    });
    DCHECK(target != nullptr);

    builder.setInsertionPoint(target);
    mlir::OperationState opstate(builder.getUnknownLoc(), "tfg.Mul");
    opstate.operands.append(target->getOperands().begin(),
                            target->getOperands().end());
    opstate.types.append(target->getResultTypes().begin(),
                         target->getResultTypes().end());
    opstate.addAttribute("T", target->getAttr("T"));
    opstate.addAttribute(dialect->getNameAttrIdentifier(),
                         builder.getStringAttr("x_times_y"));

    mlir::Operation* replacement = builder.create(opstate);
    target->replaceAllUsesWith(replacement->getResults());
    target->erase();
  }
};

// A simple testing pass that replaces the first Mul node in the module
// to a AddV2 node and names it `x_plus_y`.
struct TestPassTfDialect
    : public mlir::PassWrapper<TestPassTfDialect,
                               mlir::OperationPass<mlir::ModuleOp>> {
  TestPassTfDialect() = default;

  toolchain::StringRef getArgument() const final { return "test-pass-tf-dialect"; }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder builder(module);

    mlir::Operation* target = nullptr;
    module->walk([&target](mlir::Operation* op) {
      if (op->getName().getStringRef() == "tf.Mul") {
        target = op;
        return;
      }
    });
    DCHECK(target != nullptr);

    builder.setInsertionPoint(target);
    auto replacement = builder.create<mlir::TF::AddV2Op>(
        mlir::NameLoc::get(
            mlir::StringAttr::get(builder.getContext(), "x_plus_y")),
        target->getResultTypes(), target->getOperand(0), target->getOperand(1));
    target->replaceAllUsesWith(replacement->getResults());
    target->erase();
  }
};

inline std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfgDialectTestPass() {
  return std::make_unique<TestPassTfgDialect>();
}

inline std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfDialectTestPass() {
  return std::make_unique<TestPassTfDialect>();
}

inline void RegisterTestPass() {
  mlir::registerPass([] { return CreateTfgDialectTestPass(); });
  mlir::registerPass([] { return CreateTfDialectTestPass(); });
}

}  // namespace testing
}  // namespace function
}  // namespace core
}  // namespace machina

#endif  // MACHINA_CORE_FUNCTION_TESTING_TEST_PASS_H_
