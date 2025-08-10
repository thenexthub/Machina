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

// This file implements logic for translating mixed IR to buffer form.

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // part of Codira Toolchain

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Complex/IR/Complex.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/IR/SCF.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/IRMapping.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/ImplicitLocOpBuilder.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

struct BufferizeJITExecuteOp
    : public OpConversionPattern<tf_framework::JITExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tf_framework::JITExecuteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tf_framework::JITExecuteOp>(
        op, result_ty, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

}  // namespace

void populateExtraBufferizePatterns(ConversionTarget &target,
                                    MLIRContext *context,
                                    TypeConverter *converter,
                                    RewritePatternSet *patterns) {
  target.addLegalDialect<tf_framework::TFFrameworkDialect>();
  auto typesAreLegal = [converter](Operation *op) {
    return converter->isLegal(op->getOperandTypes()) &&
           converter->isLegal(op->getResultTypes());
  };
  target.addDynamicallyLegalOp<tf_framework::JITExecuteOp>(typesAreLegal);
  // clang-format off
  patterns->add<
      BufferizeJITExecuteOp
  >(*converter, context);
  // clang-format on
}

void populateExtraBufferizeDialects(DialectRegistry &registry) {
  registry.insert<tf_framework::TFFrameworkDialect>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
