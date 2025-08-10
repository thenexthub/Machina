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
#ifndef MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
#define MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_

#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_common {

template <typename OpTy>
mlir::LogicalResult VerifyExecuteOpCommon(OpTy op) {
  auto op_attr_array = op.getOpAttrs().getValue();
  for (auto op_attr : op_attr_array) {
    auto key_value = mlir::dyn_cast<mlir::ArrayAttr>(op_attr);
    if (!key_value || key_value.getValue().size() != 2 ||
        !mlir::isa<mlir::StringAttr>(key_value.getValue()[0]))
      return op.emitOpError() << "each op_attr should be a key-value pair, "
                                 "where the key is a string";
  }
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult VerifyFallbackExecuteOp(OpTy op) {
  auto result = VerifyExecuteOpCommon(op);
  if (failed(result)) return result;

  // Verify function attributes.
  auto op_func_attr_array = op.getOpFuncAttrs().getValue();
  for (auto op_attr : op_func_attr_array) {
    auto key_value = mlir::dyn_cast<mlir::ArrayAttr>(op_attr);
    if (!key_value || key_value.getValue().size() != 2 ||
        !mlir::isa<mlir::StringAttr>(key_value.getValue()[0]) ||
        !mlir::isa<mlir::StringAttr>(key_value.getValue()[1]))
      return op.emitOpError() << "each op_func_attr should be a key-value "
                                 "pair, where both the key and the value are "
                                 "strings";
  }
  return mlir::success();
}

template <typename OpTy>
void PrintExecuteOpFuncAttribute(mlir::OpAsmPrinter &p, OpTy op) {
  auto op_func_attrs = op.getOpFuncAttrs();
  if (!op_func_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = mlir::cast<mlir::ArrayAttr>(attr).getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << mlir::cast<mlir::StringAttr>(key).getValue();
      p << " = ";
      p << value;
    };

    auto op_func_attr_array = op_func_attrs.getValue();
    p << " {";
    toolchain::interleaveComma(op_func_attr_array, p, print_key_value);
    p << '}';
  }
}

template <typename OpTy>
void PrintExecuteOpCommon(mlir::OpAsmPrinter &p, OpTy op) {
  auto op_attrs = op.getOpAttrs();
  if (!op_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = mlir::cast<mlir::ArrayAttr>(attr).getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << mlir::cast<mlir::StringAttr>(key).getValue();
      p << " = ";
      p << value;
    };

    auto op_attr_array = op_attrs.getValue();
    p << " {";
    toolchain::interleaveComma(op_attr_array, p, print_key_value);
    p << '}';
  }
}

void GetExecuteOpAttrsCommon(
    mlir::MLIRContext *context, toolchain::ArrayRef<mlir::Attribute> op_attr_array,
    toolchain::SmallVectorImpl<std::pair<toolchain::StringRef, mlir::Attribute>>
        *op_attrs);

struct ParseExecuteOpOptions {
  bool has_chain = false;
  bool has_key = false;
  bool has_device = false;
  bool has_func_attr = false;
  bool has_cost = false;
  bool has_op_name = true;
  bool has_symbol_ref = false;
};

mlir::ParseResult ParseExecuteOpCommon(mlir::OpAsmParser &parser,
                                       mlir::Builder &builder,
                                       mlir::OperationState &result,
                                       mlir::Type tensor_type,
                                       const ParseExecuteOpOptions &options);
}  // namespace fallback_common
}  // namespace tfrt

#endif  // MACHINA_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
