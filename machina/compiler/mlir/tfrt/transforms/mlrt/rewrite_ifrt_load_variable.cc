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

#include "machina/compiler/mlir/tfrt/transforms/mlrt/rewrite_ifrt_load_variable.h"

#include <memory>
#include <vector>

#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace machina {
namespace mlrt_compiler {
namespace {

class RewriteIfrtLoadVariablePass
    : public mlir::PassWrapper<RewriteIfrtLoadVariablePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  RewriteIfrtLoadVariablePass() = default;
  RewriteIfrtLoadVariablePass &operator=(const RewriteIfrtLoadVariablePass &) =
      delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteIfrtLoadVariablePass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<machina::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  toolchain::StringRef getArgument() const final {
    return "tf-mlrt-rewrite-ifrt-load-variable";
  }

  toolchain::StringRef getDescription() const final {
    return "Convert tf.IfrtLoadVariable to tf_mlrt.TFIfrtLoadVariable";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module);

    module->walk([&](mlir::TF::IfrtLoadVariableOp load_variable_op) {
      builder.setInsertionPoint(load_variable_op);

      std::vector<mlir::Type> result_types;
      result_types.push_back(load_variable_op.getArrayKey().getType());
      result_types.push_back(builder.getType<mlrt::compiler::FutureType>());
      auto mlrt_load_variable_op =
          builder.create<tf_mlrt::TFIfrtLoadVariableOp>(
              load_variable_op->getLoc(), result_types,
              load_variable_op->getOperands(), load_variable_op->getAttrs());
      tf_mlrt::TFAwaitOp await_op;
      for (auto user : toolchain::make_early_inc_range(
               load_variable_op.getTensorFuture().getUsers())) {
        // Materialize the future for the first use. Reuse it for the rest of
        // the uses.
        if (!await_op) {
          builder.setInsertionPoint(user);
          await_op = builder.create<tf_mlrt::TFAwaitOp>(
              user->getLoc(), load_variable_op.getTensorFuture().getType(),
              mlrt_load_variable_op.getTensorFuture());
        } else {
          if (user->isBeforeInBlock(await_op)) {
            await_op->moveBefore(user);
          }
        }
        user->replaceUsesOfWith(load_variable_op.getTensorFuture(),
                                await_op.getResult());
      }

      for (auto user : toolchain::make_early_inc_range(
               load_variable_op.getArrayKey().getUsers())) {
        user->replaceUsesOfWith(load_variable_op.getArrayKey(),
                                mlrt_load_variable_op.getArrayKey());
      }

      load_variable_op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteIfrtLoadVariablePass() {
  return std::make_unique<RewriteIfrtLoadVariablePass>();
}

}  // namespace mlrt_compiler
}  // namespace machina
