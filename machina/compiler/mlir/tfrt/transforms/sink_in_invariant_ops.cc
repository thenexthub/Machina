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

#include <memory>
#include <utility>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace machina {
namespace {

// TODO(b/262610234): Generalize the sinking conditions.
// Check if the op qualifies to sink to the callee.
bool IsSinkCandidate(mlir::Operation *op) {
  return op && toolchain::isa<mlir::TF::VarHandleOp, mlir::TF::ConstOp,
                         mlir::TF::HashTableV2Op>(op);
}

// Check if the op is allowed to be sinked. We are being conservative here to
// whilelist very limited set of ops here.
struct AllowSinkHelper {
  explicit AllowSinkHelper(mlir::Operation *op, int arg_index) {
    if (toolchain::isa<mlir::TF::BatchFunctionOp,
                  mlir::TF::StatefulPartitionedCallOp>(op)) {
      allow_sink_to = true;
      callee_arg_index = arg_index;
      return;
    }

    if (toolchain::isa<mlir::TF::IfOp>(op) && arg_index > 0) {
      allow_sink_to = true;
      callee_arg_index = arg_index - 1;
      return;
    }
  }

  bool allow_sink_to = false;
  int callee_arg_index = 0;
};

toolchain::SmallVector<mlir::Value> FindValueInCallees(
    const mlir::SymbolTable &symbol_table,
    const mlir::SymbolUserMap &symbol_users, mlir::Operation *caller,
    int arg_index) {
  toolchain::SmallVector<mlir::Value> values;
  toolchain::SmallDenseSet<toolchain::StringRef> callees;
  for (const auto &named_attr : caller->getAttrs()) {
    if (auto symbol_attr =
            mlir::dyn_cast<mlir::FlatSymbolRefAttr>(named_attr.getValue())) {
      auto symbol = symbol_attr.getValue();

      auto callee = symbol_table.lookup<mlir::func::FuncOp>(symbol);
      if (!callee) continue;

      // One callee invoked by multiple caller is skipped for simplicity.
      // Consider adding support if more usage are observed from production.
      if (toolchain::ArrayRef<mlir::Operation *> users =
              symbol_users.getUsers(callee);
          users.size() > 1)
        continue;

      // Invoked by same caller multiple times, only process the first one.
      if (!callees.insert(symbol).second) continue;

      values.push_back(callee.getArgument(arg_index));
    }
  }
  return values;
}

void FindSinkTarget(
    const mlir::SymbolTable &symbol_table,
    const mlir::SymbolUserMap &symbol_users, mlir::OpResult original,
    mlir::Value value,
    toolchain::DenseMap<mlir::OpOperand *, toolchain::SmallDenseSet<mlir::OpResult>>
        &targets) {
  for (mlir::OpOperand &use : value.getUses()) {
    auto *user = use.getOwner();

    AllowSinkHelper helper(user, use.getOperandNumber());

    if (helper.allow_sink_to) {
      auto values = FindValueInCallees(symbol_table, symbol_users, user,
                                       helper.callee_arg_index);
      for (auto value : values) {
        FindSinkTarget(symbol_table, symbol_users, original, value, targets);
      }
    } else if (value != original) {
      targets[&use].insert(original);
    }
  }
}

// Sink in invariant ops like tf.Const, tf.VarHandleOp and tf.HashTableV2 ops
// into sinkable calls like tf.BatchFunction and tf.If. If there are nested
// calls, the invariant ops will only be copied at the target.
void SinkInInvariantOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  mlir::SymbolTableCollection symbol_table_collection;
  mlir::SymbolUserMap symbol_users(symbol_table_collection, module);

  // TODO(b/263191534): Replace with CallOpInterface to handle callees.
  // Identify the invariant Op, Caller, Callee FuncOp to update.

  toolchain::DenseMap<mlir::OpOperand *, toolchain::SmallDenseSet<mlir::OpResult>>
      targets;
  module.walk([&](mlir::Operation *op) {
    if (IsSinkCandidate(op)) {
      for (auto value : op->getOpResults()) {
        FindSinkTarget(symbol_table, symbol_users, value, value, targets);
      }
    }
  });

  // Clone the sinkable op associated with the func op to the func op
  mlir::OpBuilder builder(module);
  for (const auto &p : targets) {
    if (p.second.size() != 1) continue;

    auto *use = p.first;

    builder.setInsertionPointToStart(use->getOwner()->getBlock());

    mlir::OpResult original = *p.second.begin();
    auto *new_op = builder.clone(*original.getDefiningOp());

    use->get().replaceAllUsesWith(
        new_op->getResult(original.getResultNumber()));
  }
}

class SinkInInvariantOpsPass
    : public mlir::PassWrapper<SinkInInvariantOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkInInvariantOpsPass)

  toolchain::StringRef getArgument() const final {
    return "tfrt-sink-in-invariant-ops";
  }
  toolchain::StringRef getDescription() const final {
    return "Sink in the invariant ops to facilitate invariant ops hoisting.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    SinkInInvariantOps(module);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkInInvariantOpsPass() {
  return std::make_unique<SinkInInvariantOpsPass>();
}

static mlir::PassRegistration<SinkInInvariantOpsPass>
    sink_in_invariant_ops_pass;

}  // namespace machina
