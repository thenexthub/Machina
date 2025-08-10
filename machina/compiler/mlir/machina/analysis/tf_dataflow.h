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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_TF_DATAFLOW_H_
#define MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_TF_DATAFLOW_H_

#include "toolchain/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TF {

template <typename L>
class TensorflowDataflowAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<L>> {
 public:
  using StateT = dataflow::Lattice<L>;
  using dataflow::SparseForwardDataFlowAnalysis<
      StateT>::SparseForwardDataFlowAnalysis;
  using dataflow::SparseForwardDataFlowAnalysis<StateT>::getLatticeElement;
  ~TensorflowDataflowAnalysis() override = default;

  bool ForwardThroughTFOperation(Operation *op,
                                 ArrayRef<const StateT *> operands,
                                 ArrayRef<StateT *> results) {
    if (auto cast = dyn_cast<TF::CastOp>(op)) {
      this->join(results[0], *operands[0]);
    } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
      for (auto &region : while_op->getRegions()) {
        for (auto [arg, value] :
             toolchain::zip(region.getArguments(), while_op->getOperands())) {
          this->join(getLatticeElement(arg), *getLatticeElement(value));
        }
      }
    } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      func::FuncOp cond = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          while_op, while_op.getCondAttr());
      func::FuncOp body = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          while_op, while_op.getBodyAttr());
      for (auto &arg : while_op->getOpOperands()) {
        BlockArgument cond_arg = cond.getArgument(arg.getOperandNumber());
        this->join(getLatticeElement(cond_arg), *getLatticeElement(arg.get()));
        BlockArgument body_arg = body.getArgument(arg.getOperandNumber());
        this->join(getLatticeElement(body_arg), *getLatticeElement(arg.get()));
      }
    } else if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
      for (auto &arg : graph.GetFetch()->getOpOperands()) {
        if (arg.getOperandNumber() < graph.getNumResults()) {
          auto result = graph.getResult(arg.getOperandNumber());
          this->join(getLatticeElement(result), *getLatticeElement(arg.get()));
        }
      }
    } else if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
      for (auto &arg : island.GetYield()->getOpOperands()) {
        auto result = island.getResult(arg.getOperandNumber());
        this->join(getLatticeElement(result), *getLatticeElement(arg.get()));
      }
    } else {
      return false;
    }
    return true;
  }

  void setToEntryState(StateT *lattice) override {
    this->propagateIfChanged(
        lattice, lattice->join(L::EntryState(lattice->getAnchor())));
  }
};

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_TF_DATAFLOW_H_
