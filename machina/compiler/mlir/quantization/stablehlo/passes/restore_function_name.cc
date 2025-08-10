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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/xla_call_module_attrs.h"

//===----------------------------------------------------------------------===//
// The stablehlo-restore-function-name Pass.
//===----------------------------------------------------------------------===//

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_RESTOREFUNCTIONNAMEPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Restores entry function name from XlaCallModuleOp attribute.
// This restoration is required because StableHLO functions are renamed during
// the XlaCallModuleSerialization.
class RestoreFunctionNamePass
    : public impl::RestoreFunctionNamePassBase<RestoreFunctionNamePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreFunctionNamePass)

  explicit RestoreFunctionNamePass() = default;

  void runOnOperation() override;
};

void RestoreFunctionNameFromXlaCallModuleOp(TF::XlaCallModuleOp& call_op,
                                            SymbolTable& symbol_table) {
  if (!call_op->hasAttr(kOriginalStablehloEntryFunctionAttrName)) {
    return;
  }

  const auto original_function_name = call_op->getAttrOfType<StringAttr>(
      kOriginalStablehloEntryFunctionAttrName);
  const auto current_function_name = call_op->getAttrOfType<FlatSymbolRefAttr>(
      TF::kStablehloEntryFunctionAttrName);

  if (!original_function_name || !current_function_name) {
    return;
  }

  auto function =
      symbol_table.lookup<func::FuncOp>(current_function_name.getValue());
  if (function) {
    function.setName(original_function_name);
  }

  call_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                   FlatSymbolRefAttr::get(original_function_name));
}

void RestoreFunctionNamePass::runOnOperation() {
  ModuleOp module_op = getOperation();

  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);
  SymbolTable symbol_table(module_op);

  // TODO - b/298966126: Improve this logic if needed.
  module_op.walk([&](TF::XlaCallModuleOp call_op) {
    RestoreFunctionNameFromXlaCallModuleOp(call_op, symbol_table);
  });
}
}  // namespace

}  // namespace mlir::quant::stablehlo
