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

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tfrt/analysis/tensor_array_side_effect_analysis.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "machina/compiler/mlir/tfrt/transforms/utils.h"

namespace machina {
namespace tfrt_compiler {
namespace {

bool IsOpNonSideEffectingOrReadOnly(mlir::Operation* op,
                                    bool include_read_only = true) {
  // Though tf.Assert and tf.Timestamp are side-effecting, they do not
  // interfere with any other side-effecting ops. For now, if control flow
  // ops' callee functions contain them, we treat them as non-side-effecting.
  if (toolchain::isa<mlir::TF::AssertOp, mlir::TF::TimestampOp>(op)) return true;

  // TensorArray ops have explicit operand/result for handling side effects.
  if (IsTensorArrayOp(op)) return true;

  if (include_read_only) {
    if (mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op) ||
        mlir::hasSingleEffect<mlir::MemoryEffects::Read>(op) ||
        mlir::hasSingleEffect<mlir::MemoryEffects::Free>(op))
      return true;
  }

  return mlir::isMemoryEffectFree(op);
}

bool FunctionHasSideEffect(
    mlir::func::FuncOp func_op, bool include_read_only,
    toolchain::DenseMap<mlir::func::FuncOp, bool>& function_side_effect) {
  auto iter = function_side_effect.find(func_op);
  if (iter != function_side_effect.end()) return iter->second;

  auto& block = func_op.front();

  auto op_has_side_effect = [&](mlir::Operation* op) {
    if (auto while_op = toolchain::dyn_cast<mlir::TF::WhileOp>(op)) {
      if (while_op.getIsStateless()) return false;

      return FunctionHasSideEffect(while_op.cond_function(), include_read_only,
                                   function_side_effect) ||
             FunctionHasSideEffect(while_op.body_function(), include_read_only,
                                   function_side_effect);
    }

    if (auto if_op = toolchain::dyn_cast<mlir::TF::IfOp>(op)) {
      if (if_op.getIsStateless()) return false;

      return FunctionHasSideEffect(if_op.else_function(), include_read_only,
                                   function_side_effect) ||
             FunctionHasSideEffect(if_op.then_function(), include_read_only,
                                   function_side_effect);
    }

    return !IsOpNonSideEffectingOrReadOnly(op, include_read_only);
  };

  // Speculatively setting the function to have no side effect to avoid infinite
  // recursion. The correct side effect will be updated later once more
  // operations in the block are checked.
  function_side_effect[func_op] = false;

  for (mlir::Operation& op : block) {
    if (op_has_side_effect(&op)) {
      function_side_effect[func_op] = true;
      return true;
    }
  }

  function_side_effect[func_op] = false;
  return false;
}

// This pass sets `is_stateless` attribute of tf.If and tf.While ops to true if
// their callee functions contains only non-side-effecting ops.
class OptimizeTfControlFlowSideEffectPass
    : public mlir::PassWrapper<OptimizeTfControlFlowSideEffectPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OptimizeTfControlFlowSideEffectPass)

 private:
  toolchain::StringRef getArgument() const final {
    return "tfrt-optimize-tf-control-flow-side-effect";
  }
  toolchain::StringRef getDescription() const final {
    return "Set tf control flow ops to stateless if their callee functions "
           "contains only non-side-effecting ops";
  }
  void runOnOperation() override {
    auto module = getOperation();

    toolchain::SmallVector<mlir::func::FuncOp> functions;

    bool include_read_only = true;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (func.isPublic() && !IsSessionInitializer(func)) {
        functions.push_back(func);

        for (auto& op : func.front()) {
          // Skip control flow ops in the first pass.
          if (toolchain::isa<mlir::TF::WhileOp, mlir::TF::IfOp>(&op)) continue;

          if (!IsOpNonSideEffectingOrReadOnly(&op)) {
            include_read_only = false;
          }
        }
      }
    }

    toolchain::DenseMap<mlir::func::FuncOp, bool> function_side_effect;

    mlir::Builder builder(module.getContext());

    for (auto func : functions) {
      func.walk([&](mlir::Operation* op) {
        if (auto while_op = toolchain::dyn_cast<mlir::TF::WhileOp>(op)) {
          if (while_op.getIsStateless()) return;

          if (!FunctionHasSideEffect(while_op.cond_function(),
                                     include_read_only, function_side_effect) &&
              !FunctionHasSideEffect(while_op.body_function(),
                                     include_read_only, function_side_effect)) {
            while_op->setAttr("is_stateless", builder.getBoolAttr(true));
          }
        }

        if (auto if_op = toolchain::dyn_cast<mlir::TF::IfOp>(op)) {
          if (if_op.getIsStateless()) return;

          if (!FunctionHasSideEffect(if_op.else_function(), include_read_only,
                                     function_side_effect) &&
              !FunctionHasSideEffect(if_op.then_function(), include_read_only,
                                     function_side_effect)) {
            if_op->setAttr("is_stateless", builder.getBoolAttr(true));
          }
        }
      });
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateOptimizeTfControlFlowSideEffectPass() {
  return std::make_unique<OptimizeTfControlFlowSideEffectPass>();
}

static mlir::PassRegistration<OptimizeTfControlFlowSideEffectPass>
    register_pass(CreateOptimizeTfControlFlowSideEffectPass);

}  // namespace tfrt_compiler
}  // namespace machina
