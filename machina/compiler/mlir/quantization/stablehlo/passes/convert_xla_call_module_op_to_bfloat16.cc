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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir::quant::stablehlo {

absl::StatusOr<std::string> ConvertSerializedStableHloModuleToBfloat16(
    const StringRef serialized_stablehlo_module) {
  // StableHLO module is empty often because the XlaCallModuleOp is already
  // deserialized, e.g. after invoking XlaCallModuleDeserializationPass. We
  // don't handle this situation.
  if (serialized_stablehlo_module.empty()) {
    return absl::InvalidArgumentError("StableHLO module is empty.");
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> stablehlo_module_op =
      mlir::stablehlo::deserializePortableArtifact(serialized_stablehlo_module,
                                                   &context);
  auto version =
      mlir::stablehlo::getPortableArtifactVersion(serialized_stablehlo_module);
  if (failed(version)) {
    return absl::InternalError(
        "Failed to get the deserialized StableHLO version, XlaCallModuleOp "
        "must have a valid StableHLO module serialized using "
        "stablehlo::serializePortableArtifact APIs.");
  }

  // Convert the StableHLO module to bfloat16.
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(createConvertFuncToBfloat16Pass());
  if (failed(pm.run(stablehlo_module_op.get()))) {
    return absl::InternalError(
        "Failed to convert StableHLO module to bfloat16.");
  }

  std::string bytecode;
  toolchain::raw_string_ostream os(bytecode);
  if (failed(mlir::stablehlo::serializePortableArtifact(
          stablehlo_module_op.get(), version.value().toString(), os))) {
    return absl::InternalError("Failed to serialize StableHLO module.");
  }
  return bytecode;
}

#define GEN_PASS_DEF_CONVERTXLACALLMODULEOPTOBFLOAT16PASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {
class ConvertXlaCallModuleOpToBfloat16Pass
    : public impl::ConvertXlaCallModuleOpToBfloat16PassBase<
          ConvertXlaCallModuleOpToBfloat16Pass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertXlaCallModuleOpToBfloat16Pass)

  explicit ConvertXlaCallModuleOpToBfloat16Pass() = default;

 private:
  void runOnOperation() override;
};

void ConvertXlaCallModuleOpToBfloat16Pass::runOnOperation() {
  Operation* func_op = getOperation();
  SymbolTableCollection symbol_table;
  OpBuilder builder(&getContext());

  auto result = func_op->walk([&](TF::XlaCallModuleOp op) {
    // Converts the serialized StableHLO module to bfloat16.
    auto result =
        ConvertSerializedStableHloModuleToBfloat16(op.getModuleAttr());
    if (!result.ok()) {
      toolchain::errs() << "Failed to convert StableHLO module to bfloat16: "
                   << result.status().message();
      return WalkResult::interrupt();
    }
    op.setModuleAttr(StringAttr::get(&getContext(), *result));

    // Convert the `tf.XlaCallModuleOp` to bfloat16 and add casts around it.
    builder.setInsertionPoint(op);
    for (auto& op_operand : op->getOpOperands()) {
      if (quant::stablehlo::IsLargeFloatType(op_operand.get().getType())) {
        op_operand.set(builder.create<TF::CastOp>(
            op->getLoc(),
            quant::stablehlo::ToBfloat16Type(op_operand.get().getType()),
            op_operand.get()));
      }
    }
    builder.setInsertionPointAfter(op);
    for (auto op_result : op->getOpResults()) {
      if (quant::stablehlo::IsLargeFloatType(op_result.getType())) {
        const Type original_type = op_result.getType();
        op_result.setType(quant::stablehlo::ToBfloat16Type(original_type));
        const Value cast =
            builder.create<TF::CastOp>(op->getLoc(), original_type, op_result);
        op_result.replaceAllUsesExcept(cast, cast.getDefiningOp());
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace
}  // namespace mlir::quant::stablehlo
