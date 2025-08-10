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

#include <memory>
#include <string>

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfr/ir/tfr_ops.h"
#include "machina/compiler/mlir/tfr/passes/passes.h"

namespace mlir {
namespace TFR {

class RewriteQuantizedIOPass
    : public PassWrapper<RewriteQuantizedIOPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "tfr-rewrite-quantized-io"; }

  StringRef getDescription() const final {
    return "Replaces operands and results that has quantized type with their "
           "storage types.";
  }
  void runOnOperation() override;
};

void RewriteQuantizedIOPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module);
  module.walk([&](func::FuncOp func) {
    Block& block = func.front();
    Operation* terminator = block.getTerminator();

    // Replace input_arg(tensor<quant_type>) -> tfr.cast
    // with input_arg(tensor<storage_type>) -> tfr.cast
    for (BlockArgument arg : block.getArguments()) {
      Type arg_type = arg.getType();
      if (auto quant_type = toolchain::dyn_cast<quant::QuantizedType>(
              toolchain::cast<TensorType>(arg_type).getElementType())) {
        if (arg.hasOneUse() && toolchain::isa<TFR::CastOp>(*arg.user_begin())) {
          arg.setType(toolchain::cast<TensorType>(arg_type).clone(
              quant_type.getStorageType()));
        } else {
          std::string error_message;
          toolchain::raw_string_ostream os{error_message};
          os << "The argument with type ";
          arg.getType().print(os);
          os << " should have one user, which should be tfr.cast.";
          func->emitError(error_message);
          return;
        }
      }
    }

    builder.setInsertionPoint(terminator);
    // Replace tfr.cast(tensor<quant_type>) -> output
    // with tfr.cast(tensor<storage_type>) -> output
    for (OpOperand& returned_value : terminator->getOpOperands()) {
      auto returned_type =
          toolchain::dyn_cast<TensorType>(returned_value.get().getType());
      if (!returned_type ||
          !toolchain::isa<quant::QuantizedType>(returned_type.getElementType())) {
        continue;
      }

      if (auto returned_op =
              returned_value.get().getDefiningOp<TFR::CastOp>()) {
        auto new_type = returned_type.clone(
            toolchain::cast<quant::QuantizedType>(returned_type.getElementType())
                .getStorageType());
        auto new_op = TFR::CastOp::create(builder, returned_op->getLoc(),
                                          new_type, returned_op.getArg());
        returned_value.set(new_op.getResult());
        if (returned_op.use_empty()) {
          returned_op.erase();
        }
      } else {
        returned_value.get().getDefiningOp()->emitError(
            "The producer of quantized type result should be a tfr.cast op.");
        return;
      }
    }

    auto new_func_type = builder.getFunctionType(block.getArgumentTypes(),
                                                 terminator->getOperandTypes());
    func.setType(new_func_type);
  });
}

// Creates an instance of the pass to decompose the TF ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateRewriteQuantizedIOPass() {
  return std::make_unique<RewriteQuantizedIOPass>();
}

static PassRegistration<RewriteQuantizedIOPass> pass([] {
  return CreateRewriteQuantizedIOPass();
});

}  // namespace TFR
}  // namespace mlir
