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
#include <utility>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class CastBf16OpsToF32Pass
    : public PassWrapper<CastBf16OpsToF32Pass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastBf16OpsToF32Pass)
  explicit CastBf16OpsToF32Pass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-cast-bf16-ops-to-f32";
  }
  StringRef getDescription() const final {
    return "Cast BF16 operations to F32.";
  }

  void runOnOperation() override;
};

class CastBf16OpsToF32 : public RewritePattern {
 public:
  explicit CastBf16OpsToF32(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (match(op).failed()) {
      return failure();
    }
    rewrite(op, rewriter);
    return success();
  }

 private:
  LogicalResult match(Operation* op) const {
    if (isa<TF::CastOp, TF::ConstOp>(op) ||
        op->getName().hasTrait<OpTrait::ZeroOperands>()) {
      return failure();
    }
    for (Value input : op->getOperands()) {
      if (getElementTypeOrSelf(input).isBF16()) {
        return success();
      }
    }
    for (Value value : op->getResults()) {
      if (getElementTypeOrSelf(value).isBF16()) {
        return success();
      }
    }
    return failure();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const {
    // Casts inputs of the operation.
    for (int i = 0; i < op->getNumOperands(); i++) {
      Value input = op->getOperand(i);
      if (getElementTypeOrSelf(input).isBF16()) {
        Value f32_cast = rewriter.create<TF::CastOp>(
            op->getLoc(),
            CloneTypeWithNewElementType(input.getType(), rewriter.getF32Type()),
            input);
        op->setOperand(i, f32_cast);
      }
    }

    // Casts BF16 outputs of the operation.
    for (Value value : op->getResults()) {
      if (getElementTypeOrSelf(value).isBF16()) {
        value.setType(CloneTypeWithNewElementType(value.getType(),
                                                  rewriter.getF32Type()));
        rewriter.setInsertionPointAfterValue(value);
        for (Operation* user : op->getUsers()) {
          for (int i = 0; i < user->getNumOperands(); i++) {
            if (user->getOperand(i) == value) {
              Value bf16_cast = rewriter.create<TF::CastOp>(
                  user->getLoc(),
                  CloneTypeWithNewElementType(value.getType(),
                                              rewriter.getBF16Type()),
                  value);
              user->setOperand(i, bf16_cast);
            }
          }
        }
      }
    }
  }
};

#include "machina/compiler/mlir/quantization/machina/passes/cast_bf16_ops_to_f32.inc"

void CastBf16OpsToF32Pass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto module_op = getOperation();

  patterns.add<CastBf16OpsToF32>(ctx);
  populateWithGenerated(patterns);

  if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
    module_op.emitError() << "quant-cast-bf16-ops-to-f32 failed.";
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the Cast BF16 ops to F32 pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateCastBf16OpsToF32Pass() {
  return std::make_unique<CastBf16OpsToF32Pass>();
}

static PassRegistration<CastBf16OpsToF32Pass> pass;

}  // namespace quant
}  // namespace mlir
