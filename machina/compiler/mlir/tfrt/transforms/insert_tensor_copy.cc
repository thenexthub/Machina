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
#include <cassert>
#include <memory>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/compiler/stream_analysis.h"  // from @tf_runtime

namespace machina {
namespace tfrt_compiler {
namespace {

// This pass inserts copy kernels for fallback tensors when they are passed to
// multiple threads, to avoid atomic contention on their refcounts.
class InsertFallbackTensorCopy
    : public mlir::PassWrapper<InsertFallbackTensorCopy,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  }

  toolchain::StringRef getArgument() const final {
    return "tfrt-insert-fallback-tensor-copy";
  }

  toolchain::StringRef getDescription() const final {
    return "Inserts copy kernels for fallback tensors when they are passed to "
           "multiple threads, to avoid atomic contention on refcounts.";
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertFallbackTensorCopy)

  void runOnOperation() override {
    mlir::func::FuncOp func_op = getOperation();

    // Use stream analysis to know whether a value is passed to different
    // threads.
    tfrt::compiler::StreamAnalysis stream_analysis(func_op);

    auto builder = mlir::OpBuilder::atBlockBegin(&func_op.front());

    // Process function arguments first.
    for (auto arg : func_op.getArguments()) {
      if (!mlir::isa<tfrt::fallback::TFTensorType>(arg.getType())) continue;
      InsertFallbackTensorCopyForValue(arg, func_op->getLoc(), builder,
                                       stream_analysis);
    }

    // Then process each operations in the block.
    for (mlir::Operation& op : toolchain::make_early_inc_range(func_op.front())) {
      if (toolchain::isa<tfrt::fallback_async::ExecuteOp,
                    tfrt::fallback_async::ExecuteOpSeq>(&op)) {
        InsertFallbackTensorCopyForFallbackOp(&op, builder, stream_analysis);
      }
    }
  }

 private:
  void InsertFallbackTensorCopyForFallbackOp(
      mlir::Operation* op, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
    builder.setInsertionPointAfter(op);

    // Process each result value.
    for (auto result : op->getResults()) {
      if (!mlir::isa<tfrt::fallback::TFTensorType>(result.getType())) continue;
      InsertFallbackTensorCopyForValue(result, op->getLoc(), builder,
                                       stream_analysis);
    }
  }

  // Insert copy kernels to copy the result, and allocate new atomic refcount
  // if the value is going to be used by different streams/threads, in order to
  // avoid contention on the atomic counter.
  void InsertFallbackTensorCopyForValue(
      mlir::Value value, mlir::Location loc, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
    toolchain::DenseMap<int, toolchain::SmallVector<mlir::OpOperand*, 4>> stream_map;

    // Find out streams that use this value and the corresponding uses.
    for (mlir::OpOperand& use : value.getUses()) {
      // Skip return op as there should not be atomic contention on the return
      // op.
      if (toolchain::isa<tfrt::compiler::ReturnOp>(use.getOwner())) continue;

      int stream_id = stream_analysis.GetStream(use.getOwner()).id();
      stream_map[stream_id].push_back(&use);
    }

    // Organize these uses into groups. If a stream has many uses of this value,
    // put these uses into one stream. Otherwise, streams with small number
    // of uses are grouped with each other to form groups with enough uses.
    constexpr int kCopyGroupThreshold = 16;
    toolchain::SmallVector<toolchain::SmallVector<mlir::OpOperand*, 4>, 4> small_copies;
    toolchain::SmallVector<toolchain::SmallVector<mlir::OpOperand*, 4>, 4> copies;
    for (const auto& iter : stream_map) {
      if (iter.second.size() >= kCopyGroupThreshold) {
        copies.push_back(iter.second);
      } else {
        if (small_copies.empty() ||
            small_copies.back().size() >= kCopyGroupThreshold) {
          small_copies.push_back(iter.second);
        } else {
          small_copies.back().append(iter.second.begin(), iter.second.end());
        }
      }
    }

    if (!small_copies.empty())
      copies.append(small_copies.begin(), small_copies.end());

    // If it is only used by one group, then we don't need to copy.
    if (copies.size() <= 1) return;

    // Remove one group from the candidates, as we can just use the original
    // value for this group.
    copies.pop_back();

    // For each stream, we will create one new value that replaces the uses in
    // that stream.

    assert(mlir::isa<tfrt::fallback::TFTensorType>(value.getType()));

    // The number of results is the number candidate streams.
    toolchain::SmallVector<mlir::Type, 4> result_types(copies.size(),
                                                  value.getType());
    assert(!result_types.empty());

    // Create the tfrt_fallback_async.copy_if_small kernel.
    auto copy_op = builder.create<tfrt::fallback_async::CopyIfSmallOp>(
        loc, result_types, value);

    // Finally, replaces all uses with the new value.
    for (int i = 0; i < copies.size(); ++i) {
      const auto& uses = copies[i];
      auto new_value = copy_op.getResult(i);
      for (auto* use : uses) {
        use->set(new_value);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInsertFallbackTensorCopyPass() {
  return std::make_unique<InsertFallbackTensorCopy>();
}

static mlir::PassRegistration<InsertFallbackTensorCopy> register_pass(
    CreateInsertFallbackTensorCopyPass);

}  // namespace tfrt_compiler
}  // namespace machina
