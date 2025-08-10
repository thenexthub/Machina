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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_INSERTCALLONCEOPFROMSESSIONINITIALIZERPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

// This pass inserts a TFL::CallOnce op when tf_saved_model's session
// initializer is given.
class InsertCallOnceOpFromSessionInitializerPass
    : public impl::InsertCallOnceOpFromSessionInitializerPassBase<
          InsertCallOnceOpFromSessionInitializerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      InsertCallOnceOpFromSessionInitializerPass)

 private:
  void runOnOperation() override;
};

void InsertCallOnceOpFromSessionInitializerPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (func::FuncOp init_func_op :
       tf_saved_model::GetInitializerFunctions(module)) {
    for (auto func : module.getOps<func::FuncOp>()) {
      auto dict_attr =
          func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
      if (!dict_attr) continue;

      OpBuilder builder(func.getContext());
      builder.setInsertionPointToStart(&func.getBlocks().front());
      builder.create<TFL::CallOnceOp>(func.getLoc(), init_func_op.getName());
    }
  }
}

}  // namespace

// Inserts a TFL::CallOnce op when tf_saved_model's session initializer is
// given.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCallOnceOpFromSessionInitializerPass() {
  return std::make_unique<InsertCallOnceOpFromSessionInitializerPass>();
}

}  // namespace TFL
}  // namespace mlir
