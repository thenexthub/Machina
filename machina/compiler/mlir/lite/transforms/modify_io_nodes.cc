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

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_MODIFYIONODESPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

struct ModifyIONodesPass
    : public impl::ModifyIONodesPassBase<ModifyIONodesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModifyIONodesPass)

  explicit ModifyIONodesPass() {}
  explicit ModifyIONodesPass(mlir::Type input_type, mlir::Type output_type) {
    this->input_type = input_type;
    this->output_type = output_type;
  }

  void runOnOperation() override;

 private:
  // Assign the io types from the command line flag. This is only required for
  // tests.
  LogicalResult SetupInputOutputTypesIfNull(OpBuilder builder);

  // Modifies the element types of entry block arguments to be user specified
  // and returns  the new argument types.
  LogicalResult ModifyInputNodes(func::FuncOp func,
                                 toolchain::SmallVectorImpl<Type>& new_input_types,
                                 OpBuilder builder);

  // Modifies the element types of entry block returns to be user specified
  // and returns the new return types.
  LogicalResult ModifyOutputNodes(func::FuncOp func,
                                  toolchain::SmallVectorImpl<Type>& new_output_types,
                                  OpBuilder builder);

  mlir::Type input_type;
  mlir::Type output_type;
};

LogicalResult ModifyIONodesPass::SetupInputOutputTypesIfNull(
    OpBuilder builder) {
  if (input_type && output_type) return success();

  auto convert_str_to_type = [&builder](absl::string_view str) -> Type {
    if (str == "int8") {
      return builder.getIntegerType(8);
    } else if (str == "uint8") {
      return builder.getIntegerType(8, /*isSigned=*/false);
    } else if (str == "float32") {
      return builder.getF32Type();
    } else {
      return {};
    }
  };
  if (io_node_types_.size() < 2) return failure();
  if (!input_type) input_type = convert_str_to_type(io_node_types_[0]);
  if (!output_type) output_type = convert_str_to_type(io_node_types_[1]);
  return success();
}

LogicalResult ModifyIONodesPass::ModifyInputNodes(
    func::FuncOp func, toolchain::SmallVectorImpl<Type>& new_input_types,
    OpBuilder builder) {
  if (mlir::isa<FloatType>(input_type)) {
    return success();
  }

  Block& block = func.front();
  builder.setInsertionPointToStart(&block);

  for (int i = 0; i != block.getNumArguments(); ++i) {
    Value arg = block.getArgument(0);
    Type arg_type = arg.getType();
    Value new_arg = arg;
    Location loc = func.getLoc();
    if (arg.hasOneUse() && toolchain::isa<QuantizeOp>(*arg.user_begin())) {
      auto quantize_op = toolchain::cast<QuantizeOp>(*arg.user_begin());
      auto quantize_output = quantize_op.getOutput();
      auto current_type = quant::QuantizedType::getQuantizedElementType(
                              quantize_output.getType())
                              .getStorageType();
      if (current_type == input_type) {  // int8 == int8
        arg_type = quantize_output.getType();
        new_arg = block.addArgument(arg_type, loc);
        quantize_output.replaceAllUsesWith(new_arg);
      } else if (input_type.isUnsignedInteger(
                     current_type.getIntOrFloatBitWidth())) {  // int8 != uint8
        arg_type =
            ConvertSignedQuantizedToUnsigned(quantize_output.getType(), loc);
        new_arg = block.addArgument(arg_type, loc);
        quantize_op.setOperand(new_arg);
      } else {
        input_type.print(toolchain::errs() << "Requested input type ");
        quantize_op.emitError(" Couldn't be modified to the requested type.");
        return failure();
      }
      new_input_types[i] = arg_type;
      arg.dropAllUses();
      if (quantize_op.use_empty()) {
        quantize_op.erase();
      }
    } else {
      // `arg` has multiple uses or the user isn't a quantiz op (so we couldn't
      // rewrite it to a different type. Make a copy of the `arg` and replace
      // its use.
      new_arg = block.addArgument(arg_type, loc);
      arg.replaceAllUsesWith(new_arg);
    }
    block.eraseArgument(0);
  }
  return success();
}

LogicalResult ModifyIONodesPass::ModifyOutputNodes(
    func::FuncOp func, toolchain::SmallVectorImpl<Type>& new_output_types,
    OpBuilder builder) {
  Block& block = func.front();
  auto* terminator = block.getTerminator();
  builder.setInsertionPoint(terminator);

  if (mlir::isa<FloatType>(output_type)) {
    return success();
  }

  int num_return_operands = terminator->getNumOperands();
  new_output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto returned_value = terminator->getOperand(i);
    Type returned_type = returned_value.getType();
    Operation* returned_op = returned_value.getDefiningOp();
    if (returned_op && toolchain::isa<DequantizeOp>(returned_op)) {
      auto dequantize_op = toolchain::cast<DequantizeOp>(returned_op);
      auto dequantize_input = dequantize_op.getInput();
      Type current_type = quant::QuantizedType::getQuantizedElementType(
                              dequantize_input.getType())
                              .getStorageType();
      if (current_type == output_type) {  // int8 == int8
        returned_type = dequantize_input.getType();
        returned_value = dequantize_input;
      } else if (output_type.isUnsignedInteger(
                     current_type.getIntOrFloatBitWidth())) {  // int8 != uint8
        returned_type = ConvertSignedQuantizedToUnsigned(
            dequantize_input.getType(), dequantize_op.getLoc());
        // replace the dequantize op by a quantize op
        TypeAttr type_attr = TypeAttr::get(returned_type);
        auto quantize_op =
            QuantizeOp::create(builder, dequantize_op.getLoc(), returned_type,
                               dequantize_input, type_attr);
        returned_value = quantize_op.getOutput();
      } else {
        output_type.print(toolchain::errs() << "Requested output type ");
        dequantize_op.emitError(" Couldn't be modified to the requested type.");
        return failure();
      }
      new_output_types[i] = returned_type;
      terminator->setOperand(i, returned_value);
      if (dequantize_op.use_empty()) {
        dequantize_op.erase();
      }
    }
  }
  return success();
}

void ModifyIONodesPass::runOnOperation() {
  auto func = getOperation();
  auto attrs = func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");

  // Handle the entry functions only.
  if (func.getName() != "main" && (!attrs || attrs.empty())) {
    return;
  }

  OpBuilder builder(func);
  FunctionType func_type = func.getFunctionType();
  toolchain::SmallVector<Type, 4> new_input_types(func_type.getInputs().begin(),
                                             func_type.getInputs().end());
  toolchain::SmallVector<Type, 4> new_output_types(func_type.getResults().begin(),
                                              func_type.getResults().end());

  if (failed(SetupInputOutputTypesIfNull(builder))) {
    return;
  }

  if (failed(ModifyInputNodes(func, new_input_types, builder))) {
    return;
  }

  if (failed(ModifyOutputNodes(func, new_output_types, builder))) {
    return;
  }

  auto new_func_type =
      builder.getFunctionType(new_input_types, new_output_types);
  func.setType(new_func_type);
}
}  // namespace

// Creates an instance of the TensorFlow Lite modify io nodes pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateModifyIONodesPass(
    Type input_type, Type output_type) {
  return std::make_unique<ModifyIONodesPass>(input_type, output_type);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateModifyIONodesPass() {
  return std::make_unique<ModifyIONodesPass>();
}

}  // namespace TFL
}  // namespace mlir
