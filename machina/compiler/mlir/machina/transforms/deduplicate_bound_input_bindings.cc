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
#include <utility>

#include "toolchain/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {
namespace tf_saved_model {
namespace {

#define GEN_PASS_DEF_DEDUPBOUNDINPUTBINDINGPASS
#include "machina/compiler/mlir/machina/transforms/tf_savedmodel_passes.h.inc"

class DedupBoundInputBindingPass
    : public impl::DedupBoundInputBindingPassBase<DedupBoundInputBindingPass> {
  void runOnOperation() final;
};

void DedupBoundInputBindingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!mlir::tf_saved_model::IsExported(func)) return;
  toolchain::SmallDenseMap<Attribute, unsigned, 8> unique_bound_inputs;
  toolchain::BitVector arg_indices_to_erase(func.getNumArguments());
  for (unsigned i = 0, e = func.getNumArguments(); i < e; i++) {
    auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
        i, "tf_saved_model.bound_input");
    if (!attr) continue;
    auto inserted = unique_bound_inputs.insert(std::make_pair(attr, i));
    if (inserted.second) continue;
    auto duplicate_arg = func.getArgument(i);
    auto original_arg = func.getArgument(unique_bound_inputs[attr]);
    duplicate_arg.replaceAllUsesWith(original_arg);
    arg_indices_to_erase.set(i);
  }
  if (failed(func.eraseArguments(arg_indices_to_erase))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDedupBoundInputBindingPass() {
  return std::make_unique<DedupBoundInputBindingPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
