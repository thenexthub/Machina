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
#include <string>

#include "absl/strings/str_format.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_NAMEANONYMOUSITERATORSPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct NameAnonymousIteratorsPass
    : public impl::NameAnonymousIteratorsPassBase<NameAnonymousIteratorsPass> {
  void runOnOperation() override;
};

template <typename OP>
int replace(OP op, int count) {
  OpBuilder builder(op);
  std::string name = absl::StrFormat("_iterator%d", count++);

  auto new_op = builder.create<TF::IteratorOp>(
      op->getLoc(), op->getResultTypes()[0], name, /*container=*/"",
      op.getOutputTypes(), op.getOutputShapes());
  op->getResults()[0].replaceAllUsesWith(new_op->getResults()[0]);
  if (op->use_empty()) op->erase();
  return count;
}

void NameAnonymousIteratorsPass::runOnOperation() {
  int count = 1;
  getOperation().walk(
      [&](TF::AnonymousIteratorOp op) { count = replace(op, count); });
  getOperation().walk(
      [&](TF::AnonymousIteratorV2Op op) { count = replace(op, count); });
  getOperation().walk(
      [&](TF::AnonymousIteratorV3Op op) { count = replace(op, count); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateNameAnonymousIteratorsPass() {
  return std::make_unique<NameAnonymousIteratorsPass>();
}

}  // namespace TF
}  // namespace mlir
