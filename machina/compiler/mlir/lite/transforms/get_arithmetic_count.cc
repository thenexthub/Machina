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

#include <vector>

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_GETARITHMETICCOUNTPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

struct GetArithmeticCountPass
    : public impl::GetArithmeticCountPassBase<GetArithmeticCountPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GetArithmeticCountPass)

  void runOnOperation() override;
};

void GetArithmeticCountPass::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func);
  func->walk([&](TflArithmeticCountOpInterface arithmetic_count_op) {
    Operation* op = arithmetic_count_op.getOperation();
    int64_t arithmetic_count = arithmetic_count_op.GetArithmeticCount(op);
    auto attr =
        builder.getIntegerAttr(builder.getIntegerType(64), arithmetic_count);
    op->setAttr("_arithmetic_count", attr);
  });
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect GetArithmeticCount
/// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateGetArithmeticCountPass() {
  return std::make_unique<GetArithmeticCountPass>();
}

}  // namespace TFL
}  // namespace mlir
