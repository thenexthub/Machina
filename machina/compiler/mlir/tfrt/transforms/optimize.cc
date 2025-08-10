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
#include <cstdint>
#include <memory>
#include <utility>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/DenseMapInfo.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/tfrt/transforms/passes.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace tfrt_compiler {
namespace {

// Fold tf.DeviceIndex to tf.Const if it has device assigned.
class FoldDeviceIndex : public mlir::OpRewritePattern<mlir::TF::DeviceIndexOp> {
 public:
  using mlir::OpRewritePattern<mlir::TF::DeviceIndexOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::DeviceIndexOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto device = op->getAttrOfType<mlir::StringAttr>("device");
    if (!device) return mlir::failure();

    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(device.getValue().str(),
                                        &parsed_name) ||
        !parsed_name.has_type)
      return mlir::failure();

    int32_t i = 0;
    mlir::ArrayAttr device_names = op.getDeviceNames();
    for (; i < device_names.size(); ++i) {
      auto device_name =
          mlir::cast<mlir::StringAttr>(device_names[i]).getValue();
      if (device_name == parsed_name.type) break;
    }

    rewriter.replaceOpWithNewOp<mlir::TF::ConstOp>(
        op,
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get(/*shape=*/{}, rewriter.getI32Type()),
            i));

    return mlir::success();
  }
};

// A custom hash and compare function for finding out common ops.
struct SimpleOperationInfo : public toolchain::DenseMapInfo<mlir::Operation *> {
  static unsigned getHashValue(const mlir::Operation *opC) {
    return mlir::OperationEquivalence::computeHash(
        const_cast<mlir::Operation *>(opC),
        /*hashOperands=*/mlir::OperationEquivalence::directHashValue,
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const mlir::Operation *lhsC,
                      const mlir::Operation *rhsC) {
    auto *lhs = const_cast<mlir::Operation *>(lhsC);
    auto *rhs = const_cast<mlir::Operation *>(rhsC);
    if (lhs == rhs) return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return mlir::OperationEquivalence::isEquivalentTo(
        const_cast<mlir::Operation *>(lhsC),
        const_cast<mlir::Operation *>(rhsC),
        mlir::OperationEquivalence::IgnoreLocations);
  }
};

void EliminateCommonMultinomialOps(mlir::Block &block) {
  toolchain::SmallDenseMap<mlir::Operation *,
                      toolchain::SmallVector<mlir::TF::MultinomialOp>, 2,
                      SimpleOperationInfo>
      multinomial_to_eliminate;

  auto eliminate = [&]() {
    auto &list = multinomial_to_eliminate.begin()->second;
    auto first = list.front();
    for (auto op : toolchain::drop_begin(list)) {
      op.getOutput().replaceAllUsesWith(first.getOutput());
      op->erase();
    }
    multinomial_to_eliminate.clear();
  };

  for (auto &op : block) {
    auto multinomial_op = toolchain::dyn_cast<mlir::TF::MultinomialOp>(&op);
    // Conservatively, we only eliminate back-to-back tf.Multinomial ops.
    if (multinomial_op) {
      if (multinomial_to_eliminate.find(multinomial_op) ==
              multinomial_to_eliminate.end() &&
          !multinomial_to_eliminate.empty()) {
        // If the current op is a tf.Multinomial but it is different from the
        // preiously found tf.Multinomial, then we eliminate the prviously found
        // tf.Multinomial.
        eliminate();
      }
      multinomial_to_eliminate[multinomial_op].push_back(multinomial_op);
    } else if (!multinomial_to_eliminate.empty()) {
      // If the current op is not a tf.Multinomial, then we eliminate previously
      // found tf.Multinomial
      eliminate();
    }
  }
}

// Optimization pass for TFRT-specific rewrite patterns.
class OptimizeTfForTfrt
    : public mlir::PassWrapper<OptimizeTfForTfrt,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTfForTfrt)

  toolchain::StringRef getArgument() const final { return "optimize-tf-for-tfrt"; }

  toolchain::StringRef getDescription() const final {
    return "optmize TF MLIR for TFRT workflow.";
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::RewritePatternSet pattern_list(context);
    pattern_list.add<FoldDeviceIndex>(context);
    patterns_ = std::move(pattern_list);
    return mlir::success();
  }

  void runOnOperation() override {
    auto func = getOperation();

    EliminateCommonMultinomialOps(func.getBody().front());

    if (mlir::failed(mlir::applyPatternsGreedily(func, patterns_)))
      signalPassFailure();
  }

 private:
  mlir::FrozenRewritePatternSet patterns_;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateOptimizeTfForTfrtPass() {
  return std::make_unique<OptimizeTfForTfrt>();
}

static mlir::PassRegistration<OptimizeTfForTfrt> register_pass;

}  // namespace tfrt_compiler
}  // namespace machina
