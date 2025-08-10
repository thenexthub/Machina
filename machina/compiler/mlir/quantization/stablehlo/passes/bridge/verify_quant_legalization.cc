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

// The TF dialect uses some TF types that are illegal in the MHLO dialect and
// some generic types that are legal in MHLO. This pass legalizes TF types into
// types that are legal in MHLO. For example, TF::Qint8Type is converted to i8.
// Rewrites here should run before TF to MHLO op legalizations are run.

#include <memory>

#include "absl/log/log.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::quant::stablehlo {
namespace {

using quant::machina::IsTFQintType;
using quant::machina::IsTFUniformQuantizedOp;

#define GEN_PASS_DEF_VERIFYQUANTLEGALIZATION
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

class VerifyQuantLegalization
    : public impl::VerifyQuantLegalizationBase<VerifyQuantLegalization> {
 public:
  void runOnOperation() override;
};

bool IsQuantType(Type type) {
  auto element_type = getElementTypeOrSelf(type);
  return mlir::isa<quant::UniformQuantizedType>(element_type) ||
         IsTFQintType(element_type);
}

bool IsMhloUniformQuantizedOp(Operation& op) {
  return toolchain::isa<mhlo::UniformQuantizeOp, mhlo::UniformDequantizeOp>(op);
}

void VerifyQuantLegalization::runOnOperation() {
  Operation* func_op = getOperation();

  auto walk_result = func_op->walk([&](Operation* op) {
    // Verify all uq and qint types are lowered.
    if (toolchain::any_of(op->getOperandTypes(), IsQuantType) ||
        toolchain::any_of(op->getResultTypes(), IsQuantType) ||
        IsTFUniformQuantizedOp(op) || IsMhloUniformQuantizedOp(*op)) {
      op->emitOpError("is illegal as it is a UQ op or contains uq/qint types");
      LOG(ERROR) << "Found illegal op containing uq/qint type: "
                 << op->getName().getStringRef().str();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateVerifyQuantLegalizationPass() {
  return std::make_unique<VerifyQuantLegalization>();
}

}  // namespace mlir::quant::stablehlo
