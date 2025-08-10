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

#include "machina/compiler/mlir/machina/analysis/resource_dataflow.h"

#include "toolchain/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // part of Codira Toolchain
#include "mlir/Analysis/DataFlowFramework.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/tf_dataflow.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

#define DEBUG_TYPE "resource-dataflow"

namespace mlir {
namespace TF {

namespace {
constexpr char kCompositeDevice[] = "tf._composite_device";
}  // namespace

ResourceConstructingOps::ResourceConstructingOps(Operation *op) {
  if (op) ops.insert(op);
}

ResourceConstructingOps ResourceConstructingOps::EntryState(
    MLIRContext *context) {
  return ResourceConstructingOps();
}
ResourceConstructingOps ResourceConstructingOps::EntryState(Value value) {
  if (auto barg = mlir::dyn_cast<BlockArgument>(value)) {
    if (func::FuncOp func =
            dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
      SymbolTable symbol_table(func->getParentOfType<ModuleOp>());
      auto global_tensor = tf_saved_model::LookupBoundInputOfType<
          tf_saved_model::GlobalTensorOp>(func, barg.getArgNumber(),
                                          symbol_table);
      ResourceConstructingOps result(global_tensor);
      return result;
    }
  } else if (auto vh = dyn_cast<TF::VarHandleOp>(value.getDefiningOp())) {
    return ResourceConstructingOps(vh);
  } else if (auto it = dyn_cast<TF::IteratorOp>(value.getDefiningOp())) {
    return ResourceConstructingOps(it);
  }
  return ResourceConstructingOps();
}

ResourceConstructingOps ResourceConstructingOps::join(
    const ResourceConstructingOps &lhs, const ResourceConstructingOps &rhs) {
  // Take union of both sets of possible GlobalTensorOp values that can be
  // referenced here.
  ResourceConstructingOps ret;
  ret.ops.insert(lhs.ops.begin(), lhs.ops.end());
  ret.ops.insert(rhs.ops.begin(), rhs.ops.end());
  return ret;
}

void ResourceConstructingOps::print(raw_ostream &os) const {
  toolchain::interleaveComma(ops, os << "[");
  os << "]";
}

IsComposite::IsComposite(Operation *op) {}

IsComposite IsComposite::EntryState(MLIRContext *context) {
  return IsComposite();
}

IsComposite IsComposite::EntryState(Value value) {
  IsComposite result;
  if (auto barg = mlir::dyn_cast<BlockArgument>(value)) {
    if (func::FuncOp func =
            dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
      if (func.getArgAttr(barg.getArgNumber(), kCompositeDevice)) {
        result.is_on_composite_device = true;
      }
      return result;
    }
  }
  return result;
}

IsComposite IsComposite::join(const IsComposite &lhs, const IsComposite &rhs) {
  IsComposite ret;
  ret.is_on_composite_device =
      lhs.is_on_composite_device || rhs.is_on_composite_device;
  return ret;
}

void IsComposite::print(raw_ostream &os) const {
  if (is_on_composite_device) {
    os << "COMPOSITE";
  } else {
    os << "NOT_COMPOSITE";
  }
}

class ResourceDataflowAnalysis
    : public TensorflowDataflowAnalysis<ResourceConstructingOps> {
 public:
  using TensorflowDataflowAnalysis<
      ResourceConstructingOps>::TensorflowDataflowAnalysis;
  LogicalResult visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                               ArrayRef<StateT *> results) override {
    if (ForwardThroughTFOperation(op, operands, results))
      return mlir::success();
    setAllToEntryStates(results);
    return mlir::success();
  }
  ~ResourceDataflowAnalysis() override = default;
};

class IsCompositeDataflowAnalysis
    : public TensorflowDataflowAnalysis<IsComposite> {
 public:
  using TensorflowDataflowAnalysis<IsComposite>::TensorflowDataflowAnalysis;
  LogicalResult visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                               ArrayRef<StateT *> results) override {
    if (ForwardThroughTFOperation(op, operands, results))
      return mlir::success();
    setAllToEntryStates(results);
    return mlir::success();
  }
  ~IsCompositeDataflowAnalysis() override = default;
};

void LoadResourceDataflowAnalysis(DataFlowSolver &solver) {
  solver.load<ResourceDataflowAnalysis>();
}

void LoadIsCompositeDataflowAnalysis(DataFlowSolver &solver) {
  solver.load<IsCompositeDataflowAnalysis>();
}

}  // namespace TF
}  // namespace mlir
