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
#include <string>

#include "absl/strings/str_cat.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // part of Codira Toolchain
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // part of Codira Toolchain
#include "mlir/Analysis/DataFlowFramework.h"  // part of Codira Toolchain
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/resource_dataflow.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {

namespace {

std::string GetVariableName(Operation* op) {
  if (auto handle = dyn_cast<TF::VarHandleOp>(op)) {
    std::string container = handle.getContainer().str();
    std::string shared_name = handle.getSharedName().str();
    if (container.empty()) {
      return absl::StrCat("vars.", shared_name);
    } else {
      return absl::StrCat("vars.", container, ".", shared_name);
    }
  } else if (auto global = dyn_cast<tf_saved_model::GlobalTensorOp>(op)) {
    return absl::StrCat("vars.", global.getSymName().str());
  }
  return "<no name>";
}

Operation* GetHandleSource(Operation* op, DataFlowSolver& solver) {
  Value resource;
  if (auto read = toolchain::dyn_cast<TF::ReadVariableOp>(op)) {
    resource = read.getResource();
  } else if (auto write = toolchain::dyn_cast<TF::AssignVariableOp>(op)) {
    resource = write.getResource();
  }
  const TF::ResourceDataflowState* state =
      solver.lookupState<TF::ResourceDataflowState>(resource);
  if (!state) {
    return nullptr;
  }
  auto ops = state->getValue().ops;
  if (ops.size() != 1) {
    return nullptr;
  }
  Operation* source = *ops.begin();
  return source;
}

Attribute GetInitialValue(Operation* source) {
  if (auto global = dyn_cast<tf_saved_model::GlobalTensorOp>(source)) {
    if (global.getValue()) {
      return *global.getValue();
    }
  }
  return nullptr;
}

Type GetGlobalType(Operation* source) {
  if (auto var_handle_op = dyn_cast<TF::VarHandleOp>(source)) {
    // Resources are represented as tensor<resource<tensor<...>>>, so
    // unwrap until we get to the inner tensor<...>.
    auto tensor =
        toolchain::dyn_cast<TensorType>(var_handle_op.getResource().getType());
    if (!tensor) return nullptr;
    TF::ResourceType resource =
        toolchain::dyn_cast<TF::ResourceType>(tensor.getElementType());
    if (!resource || resource.getSubtypes().size() != 1) return nullptr;
    return resource.getSubtypes().front();
  } else if (auto global_tensor_op =
                 dyn_cast<tf_saved_model::GlobalTensorOp>(source)) {
    return global_tensor_op.getType();
  }
  // Likely can't actually happen, assuming tf_saved_model.semantics checks
  // already ran.
  return nullptr;
}

ml_program::GlobalOp CreateGlobalOpFromOp(Operation* source, OpBuilder& builder,
                                          SymbolTable& symbol_table) {
  Type type = GetGlobalType(source);
  std::string name = GetVariableName(source);
  if (auto existing = symbol_table.lookup<ml_program::GlobalOp>(name)) {
    // This might be of a different type, but we'll do a Cast later.
    return existing;
  }

  Attribute initial_value = GetInitialValue(source);
  if (!initial_value) {
    initial_value = builder.getZeroAttr(type);
    if (!initial_value) {
      initial_value = builder.getArrayAttr({});
    }
  }

  if (!type) return nullptr;

  auto globalOp = builder.create<ml_program::GlobalOp>(
      builder.getBlock()->getParentOp()->getLoc(), name, type, false,
      initial_value, nullptr);
  symbol_table.insert(globalOp);

  return globalOp;
}

}  // namespace

#define GEN_PASS_DEF_LOWERVARIABLEOPSTOMLPROGRAMPASS
#include "machina/compiler/mlir/machina/transforms/tf_savedmodel_passes.h.inc"

struct LowerVariableOpsToMlProgramPass
    : public impl::LowerVariableOpsToMlProgramPassBase<
          LowerVariableOpsToMlProgramPass> {
  explicit LowerVariableOpsToMlProgramPass() = default;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    ml_program::MLProgramDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    TF::LoadResourceDataflowAnalysis(solver);
    if (failed(solver.initializeAndRun(module))) return signalPassFailure();

    SymbolTable symbol_table(module);

    OpBuilder globalBuilder(module.getBodyRegion());

    module.walk([&](TF::ReadVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp =
          CreateGlobalOpFromOp(source, globalBuilder, symbol_table);
      if (!globalOp) return;
      OpBuilder builder(op);
      Operation* load = builder.create<mlir::ml_program::GlobalLoadOp>(
          op.getLoc(), globalOp.getType(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()));
      if (globalOp.getType() != op.getValue().getType()) {
        load = builder.create<TF::CastOp>(op.getLoc(), op.getValue().getType(),
                                          load->getResult(0));
      }
      op.getResult().replaceAllUsesWith(load->getResult(0));
      op.erase();
    });

    module.walk([&](TF::AssignVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp =
          CreateGlobalOpFromOp(source, globalBuilder, symbol_table);
      if (!globalOp) return;
      symbol_table.insert(globalOp);
      OpBuilder builder(op);
      globalOp.setIsMutableAttr(builder.getUnitAttr());
      Value value_to_store = op.getValue();
      if (globalOp.getType() != op.getValue().getType()) {
        value_to_store = builder.create<TF::CastOp>(
            op.getLoc(), globalOp.getType(), value_to_store);
      }
      builder.create<mlir::ml_program::GlobalStoreOp>(
          op.getLoc(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()),
          value_to_store);
      op.erase();
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
CreateLowerVariableOpsToMlProgramPass() {
  return std::make_unique<LowerVariableOpsToMlProgramPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
