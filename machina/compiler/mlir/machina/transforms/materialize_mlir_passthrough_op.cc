/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include <iterator>
#include <memory>
#include <string>
#include <tuple>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

#define DEBUG_TYPE "tf-materialize-passthrough-op"

namespace mlir {
namespace {

#define GEN_PASS_DEF_MATERIALIZEPASSTHROUGHOP
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class MaterializePassthroughOpPass
    : public impl::MaterializePassthroughOpBase<MaterializePassthroughOpPass> {
 public:
  void runOnOperation() override;
};

void MaterializePassthroughOpPass::runOnOperation() {
  getOperation().walk([](TF::MlirPassthroughOp op) {
    std::string module_string(op.getMlirModule());
    // Parse the module.
    auto nested_module =
        parseSourceString<ModuleOp>(module_string, op.getContext());
    if (!nested_module) {
      op->emitError() << "could not parse attached MLIR module";
      return;
    }
    func::FuncOp main =
        dyn_cast<func::FuncOp>(nested_module->lookupSymbol("main"));
    if (!main) {
      op->emitError() << "MLIR Opaque Op expects a main() entry point\n";
      return;
    }
    if (main.getNumArguments() != op->getNumOperands()) {
      op->emitError() << "mismatch between MLIR Opaque Op number of operands ("
                      << op->getNumOperands()
                      << ") and main() entry point in the module ("
                      << main.getNumArguments() << " args)\n";
      return;
    }
    if (main.getFunctionType().getNumResults() != op->getNumResults()) {
      op->emitError() << "mismatch between MLIR Opaque Op number of results ("
                      << op->getNumResults()
                      << ") and main() entry point in the module ("
                      << main.getFunctionType().getNumResults()
                      << " results)\n";
      return;
    }
    Region &body = main.getBody();
    if (!toolchain::hasSingleElement(body)) {
      op->emitError() << "MLIR Opaque Op expects a main() entry point with a "
                         "single block\n";
      return;
    }
    Block &block = body.front();
    for (const auto &arg_mapping :
         toolchain::zip(block.getArguments(), op->getOperands())) {
      std::get<0>(arg_mapping).replaceAllUsesWith(std::get<1>(arg_mapping));
    }
    op->getBlock()->getOperations().splice(op->getIterator(),
                                           block.getOperations(), block.begin(),
                                           std::prev(block.end()));
    Operation &return_op = block.front();
    for (auto ret_mapping :
         toolchain::zip(op->getResults(), return_op.getOperands())) {
      std::get<0>(ret_mapping).replaceAllUsesWith(std::get<1>(ret_mapping));
    }
    op->erase();
  });
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<func::FuncOp>>
CreateMaterializePassthroughOpPass() {
  return std::make_unique<MaterializePassthroughOpPass>();
}
}  // namespace TF

}  // namespace mlir
