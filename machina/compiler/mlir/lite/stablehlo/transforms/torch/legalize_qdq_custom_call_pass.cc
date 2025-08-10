/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "mlir/AsmParser/AsmParser.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
#define GEN_PASS_DEF_LEGALIZEQDQCUSTOMCALLPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

// Quant dialect is not registered in the Python MLIR pybinding used by
// odml-torch. Therefore, stablehlo.uniform_quantize/uniform_dequantize ops and
// quant types are lowered as stablehlo.custom_call to pass MLIR verification
// and VHLO serialization before converter. This pass reconstruct Q/DQ ops and
// quant types from those custom_call ops.

class LegalizeQDQCustomCallPass
    : public impl::LegalizeQDQCustomCallPassBase<LegalizeQDQCustomCallPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeQDQCustomCallPass);

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    getOperation()->walk([&](stablehlo::CustomCallOp op) {
      toolchain::StringRef target_name = op.getCallTargetNameAttr().strref();
      if (target_name == "odml_torch.uniform_quantize") {
        auto quant_result_ty =
            mlir::dyn_cast<mlir::StringAttr>(op.getBackendConfigAttr());
        if (!quant_result_ty) {
          return;
        }
        builder.setInsertionPointAfter(op);
        auto q_op = builder.create<stablehlo::UniformQuantizeOp>(
            op.getLoc(), mlir::parseType(quant_result_ty, context),
            op.getOperand(0));
        op.getResult(0).replaceAllUsesWith(q_op.getResult());
        op.erase();
      } else if (target_name == "odml_torch.uniform_dequantize") {
        builder.setInsertionPointAfter(op);
        auto dq_op = builder.create<stablehlo::UniformDequantizeOp>(
            op.getLoc(), op.getResult(0).getType(), op.getOperand(0));
        op.getResult(0).replaceAllUsesWith(dq_op.getResult());
        op.erase();
      }
    });
  }
};

}  // namespace
}  // namespace odml
}  // namespace mlir
