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

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/transforms/tf_saved_model_passes.h"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_CONVERTSESSIONINITIALIZERTOFUNCTIONPASS
#include "machina/compiler/mlir/machina/transforms/tf_savedmodel_passes.h.inc"

struct ConvertSessionInitializerToFunctionPass
    : public impl::ConvertSessionInitializerToFunctionPassBase<
          ConvertSessionInitializerToFunctionPass> {
  void runOnOperation() override;
};

void ConvertSessionInitializerToFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto session_initializer = tf_saved_model::GetSessionInitializerOp(module);
  if (!session_initializer) return;

  OpBuilder builder(session_initializer);
  const char *name = "session_initializer";

  // In the (unlikely) case of there already being a session initializer
  // function, bail out.
  if (SymbolTable::lookupSymbolIn(module, name)) {
    module->emitWarning("session_initializer function already exists");
    session_initializer.erase();
    return;
  }

  auto init = builder.create<func::FuncOp>(
      module.getLoc(), name,
      FunctionType::get(module.getContext(), /*inputs=*/{}, /*results=*/{}));

  // Make savedmodel verification happy.
  init->setAttr("tf_saved_model.exported_names",
                builder.getStrArrayAttr({name}));

  builder.setInsertionPointToStart(init.addEntryBlock());

  for (func::FuncOp func : tf_saved_model::GetInitializerFunctions(module)) {
    if (func.getNumArguments() != 0) {
      session_initializer->emitWarning(
          "encountered session initializers with arguments");
      continue;
    }

    // Since we're now calling this function, savedmodel verification
    // needs it to be private.
    func.setVisibility(SymbolTable::Visibility::Private);
    func->removeAttr("tf_saved_model.exported_names");

    ArrayRef<Value> args;
    builder.create<func::CallOp>(session_initializer.getLoc(),
                                 func.getFunctionType().getResults(),
                                 func.getSymName(), args);
  }
  builder.create<func::ReturnOp>(session_initializer.getLoc());

  session_initializer.erase();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateConvertSessionInitializerToFunctionPass() {
  return std::make_unique<ConvertSessionInitializerToFunctionPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
