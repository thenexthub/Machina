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

#include "machina/dtensor/mlir/op_utils.h"

#include <cstdint>
#include <optional>
#include <string>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/Hashing.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"

namespace machina {
namespace dtensor {

// OpHash prints the `op` into a string and performs hash value on the output
// string.
//
// The `print` includes the full representation of the `op`, e.g., target
// register, layout, shape, etc. This should be sufficient to uniquely
// identify the operation in most cases. This does not capture function scope
// (identical op in 2 separate functions).
uint64_t OpHash(mlir::Operation* op) {
  std::string output;
  toolchain::raw_string_ostream output_stream(output);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(1024);
  op->print(output_stream, flags);
  return toolchain::hash_value(output);
}

// Returns FuncOp if `op` is a callable.
std::optional<mlir::func::FuncOp> MaybeFindFunction(mlir::Operation* op) {
  auto call_op = toolchain::dyn_cast<mlir::CallOpInterface>(op);
  if (!call_op) return std::nullopt;

  mlir::CallInterfaceCallable callable = call_op.getCallableForCallee();
  mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
  if (!sym) return std::nullopt;

  mlir::func::FuncOp func = toolchain::dyn_cast<mlir::func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func) return std::nullopt;

  return func;
}

void RemoveDTensorLayoutOps(mlir::ModuleOp module,
                            bool remove_xla_spmd_layouts) {
  toolchain::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk([&](mlir::TF::DTensorLayout layout) {
    // Remove layout ops only for layouts running on DTensor SPMD.
    // Layout ops will be preserved for XLA SPMD to annotate sharding
    // later down the DTensor stack.
    if (remove_xla_spmd_layouts || !layout.getLayout().mesh().use_xla_spmd()) {
      layout_ops.emplace_back(layout);
    }
  });

  for (auto layout_op : layout_ops) {
    layout_op.getOutput().replaceAllUsesWith(layout_op.getInput());
    layout_op.erase();
  }
}

mlir::LogicalResult ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(
    mlir::ModuleOp module) {
  toolchain::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk([&](mlir::TF::DTensorLayout op) { layout_ops.emplace_back(op); });

  toolchain::DenseSet<mlir::TF::DTensorLayout> deleted_layout_ops;

  for (auto layout_op : toolchain::reverse(layout_ops)) {
    if (deleted_layout_ops.contains(layout_op)) {
      continue;
    }
    while (auto input_layout_op =
               toolchain::dyn_cast_or_null<mlir::TF::DTensorLayout>(
                   layout_op.getInput().getDefiningOp())) {
      // Check that layout of input DTensorLayout op is equivalent to
      // the layout of its connected DTensorLayout op.
      if (layout_op.getLayout() != input_layout_op.getLayout()) {
        return layout_op.emitOpError(
            "Found inconsistent layout. This should never happen.");
      }

      // Replace DTensorLayout op with identity op.
      mlir::OpBuilder builder(input_layout_op);
      auto new_identity = builder.create<mlir::TF::IdentityOp>(
          input_layout_op->getLoc(), input_layout_op.getType(),
          input_layout_op.getInput());
      input_layout_op.getOutput().replaceAllUsesWith(new_identity.getOutput());
      input_layout_op.erase();

      deleted_layout_ops.insert(input_layout_op);
    }
  }

  return mlir::success();
}

// For all constants with multiple usages, clone the constants so that each
// constant operation has at most 1 usage.
void DuplicateConstants(mlir::Operation* op) {
  toolchain::SmallVector<mlir::TF::ConstOp, 4> const_ops;
  op->walk(
      [&](mlir::TF::ConstOp const_op) { const_ops.emplace_back(const_op); });

  for (mlir::TF::ConstOp const_op : const_ops) {
    mlir::OpBuilder builder(const_op);
    auto uses = const_op->getUses();
    if (uses.empty()) return;

    toolchain::SmallDenseMap<mlir::Operation*, mlir::OpOperand*> const_use_map;
    mlir::OpOperand& first_use = *uses.begin();
    for (mlir::OpOperand& use : uses) {
      if (&use == &first_use) continue;

      mlir::Operation* new_const = builder.clone(*const_op);
      const_use_map.try_emplace(new_const, &use);
    }

    for (const auto& it : const_use_map) it.second->set(it.first->getResult(0));
  }
}

std::string GetOperationName(mlir::ModuleOp module) {
  auto operation_name_attr =
      module->getAttrOfType<mlir::StringAttr>(kEagerOperationName);
  const std::string operation_name =
      operation_name_attr ? operation_name_attr.getValue().str() : "unknown";

  return operation_name;
}
}  // namespace dtensor
}  // namespace machina
