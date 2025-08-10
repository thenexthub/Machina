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

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Region.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/xla_call_module_attrs.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_UNWRAPXLACALLMODULEOPPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Unwraps XlaCallModule ops without quantizable trait that call function with
// '_from_xla_call_module' trait.
class UnwrapXlaCallModuleOpPass
    : public impl::UnwrapXlaCallModuleOpPassBase<UnwrapXlaCallModuleOpPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnwrapXlaCallModuleOpPass)

  explicit UnwrapXlaCallModuleOpPass() = default;

 private:
  void runOnOperation() override;
};

void UnwrapXlaCallModuleOp(TF::XlaCallModuleOp call_op,
                           SymbolTable& symbol_table) {
  // Do not inline lifted quantized functions used for fusing patterns.
  // TODO - b/310539922: Remove reference to TF/TFL utils.
  if (call_op->hasAttr(kQuantTraitAttrName)) {
    return;
  }

  auto function_name = call_op
                           ->getAttrOfType<FlatSymbolRefAttr>(
                               TF::kStablehloEntryFunctionAttrName)
                           .getValue();
  func::FuncOp func_op = symbol_table.lookup<func::FuncOp>(function_name);

  // We should not unwrap if the function is not from
  // ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass.
  if (!func_op->hasAttr(TF::kFromXlaCallModuleAttrName)) {
    return;
  }

  MLIRContext* context = call_op.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointAfter(call_op);

  IRMapping arg_mapper;
  bool call_op_has_platform_index_arg = call_op.getPlatforms().size() > 1;
  // Add an argument for platform_index. This allows for multiple platforms.
  // TODO: b/310291615 - find a better way for multi-platform support.
  if (call_op_has_platform_index_arg) {
    arg_mapper.map(func_op.getArgument(0),
                   builder.create<mhlo::ConstantOp>(
                       func_op.getLoc(), builder.getI16IntegerAttr(0)));
  }
  for (auto [func_arg, operand] : toolchain::zip_equal(
           func_op.getArguments().take_back(call_op.getNumOperands()),
           call_op.getOperands())) {
    arg_mapper.map(func_arg, operand);
  }

  Region& function_body = func_op.getBody();
  IRMapping new_op_mapper;
  for (Operation& op : function_body.getOps()) {
    if (toolchain::isa<func::ReturnOp>(op)) {
      for (auto [call_result, return_value] :
           toolchain::zip_equal(call_op.getResults(), op.getOperands())) {
        Value new_result = new_op_mapper.lookup(return_value);

        call_result.replaceAllUsesWith(new_result);
      }
      continue;
    }

    Operation& new_op = *builder.clone(op, arg_mapper);
    for (auto [result, new_result] :
         toolchain::zip_equal(op.getResults(), new_op.getResults())) {
      new_op_mapper.map(result, new_result);
    }
  }

  call_op.erase();
}

void UnwrapXlaCallModuleOpPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  SymbolTable symbol_table(module_op);

  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    Region& function_body = func_op.getBody();

    function_body.walk([&](TF::XlaCallModuleOp call_op) {
      UnwrapXlaCallModuleOp(call_op, symbol_table);
    });
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo
