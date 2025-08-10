/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_executor::GraphOp;
using ::mlir::tf_executor::IslandOp;

constexpr StringRef kSharedNameAttr = "shared_name";

class MergeDuplicateResourceOpsPass
    : public PassWrapper<MergeDuplicateResourceOpsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeDuplicateResourceOpsPass)

  StringRef getArgument() const final {
    return "quant-merge-duplicate-resource-ops";
  }

  StringRef getDescription() const final {
    return "Merge resource ops that have the same shared name.";
  }

  void runOnOperation() override;
};

// Checks if the island op contains a resource op like Variable or Hashtable
// and returns that resource op. Otherwise, returns null.
Operation* GetResourceOp(Operation* op) {
  // Check if the island has only one block thats contain two ops, including
  // one resource op and one Yield op.
  auto island_op = toolchain::dyn_cast_or_null<IslandOp>(op);
  if (!island_op || !island_op.getBody().hasOneBlock()) return nullptr;
  auto& island_block = island_op.getBody().front();
  if (++island_block.begin() != --island_block.end()) return nullptr;

  Operation* resource_op = &island_block.front();
  if (toolchain::isa<TF::VarHandleOp, TF::HashTableOp, TF::HashTableV2Op,
                TF::MutableHashTableV2Op>(resource_op)) {
    return resource_op;
  }
  return nullptr;
}

// Returns the `shared_name` attribute value if exists. If not, returns an
// empty string.
StringRef GetSharedName(Operation* op) {
  if (!op->hasAttrOfType<StringAttr>(kSharedNameAttr)) return "";
  return op->getAttrOfType<StringAttr>(kSharedNameAttr).getValue();
}

// Gets the GraphOp from the function op. Returns an empty op iff it doesn't
// exist.
// TODO(b/284222084): Move executor dialect utilities to a new library.
GraphOp GetGraphOpFromFuncOp(func::FuncOp func_op) {
  if (func_op->getNumRegions() == 0 || func_op.getBody().empty()) return {};

  auto graph_op_range = func_op.front().without_terminator();
  if (toolchain::hasSingleElement(graph_op_range)) {
    // The pass runs on a valid tf_executor dialect, so the op should be the
    // GraphOp.
    return cast<GraphOp>(graph_op_range.begin());
  }

  return {};
}

void MergeDuplicateResourceOpsPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  GraphOp graph_op = GetGraphOpFromFuncOp(func_op);
  if (!graph_op) return;

  toolchain::StringMap<Operation*> shared_name_to_resource;
  toolchain::SmallVector<Operation*> ops_to_remove;
  for (Operation& op : graph_op.GetBody().without_terminator()) {
    Operation* resource_op = GetResourceOp(&op);
    if (!resource_op) continue;
    StringRef shared_name = GetSharedName(resource_op);
    if (shared_name.empty()) continue;

    if (!shared_name_to_resource.contains(shared_name)) {
      shared_name_to_resource[shared_name] = resource_op;
      continue;
    }

    auto existing_resource = shared_name_to_resource[shared_name];
    if (resource_op->getName().getStringRef() !=
            existing_resource->getName().getStringRef() ||
        resource_op->getResult(0).getType() !=
            existing_resource->getResult(0).getType()) {
      resource_op->emitOpError(
          "This op has the same `shared_name` but different type with another "
          "resource op in the function");
      signalPassFailure();
      return;
    }
    op.replaceAllUsesWith(existing_resource->getParentOp()->getResults());
    ops_to_remove.push_back(&op);
  }

  // Remove op after the loop to avoid crash.
  for (Operation* op : ops_to_remove) {
    op->erase();
  }
}

static PassRegistration<MergeDuplicateResourceOpsPass> pass{};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateMergeDuplicateResourceOpsPass() {
  return std::make_unique<MergeDuplicateResourceOpsPass>();
}

}  // namespace quant
}  // namespace mlir
