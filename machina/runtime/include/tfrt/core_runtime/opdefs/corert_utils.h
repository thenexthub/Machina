/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares utils for both 'corert' and 'corert_sync'dialect.
#ifndef TFRT_CORE_RUNTIME_OPDEFS_CORERT_UTILS_H_
#define TFRT_CORE_RUNTIME_OPDEFS_CORERT_UTILS_H_

#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace tfrt {
namespace corert {

template <typename OpTy>
LogicalResult VerifyExecuteOpImpl(OpTy op) {
  auto op_attr_array = op.getOpAttrs().getValue();
  for (auto op_attr : op_attr_array) {
    auto key_value = mlir::dyn_cast<ArrayAttr>(op_attr);
    if (!key_value || key_value.getValue().size() != 2 ||
        !mlir::isa<StringAttr>(key_value.getValue()[0]))
      return op.emitOpError() << "each op_attr should be a key-value pair, "
                                 "where the key is a string";
  }
  return success();
}

template <typename OpTy>
void PrintExecuteOpFuncAttribute(mlir::OpAsmPrinter &p, OpTy op) {
  auto op_func_attrs = op.getOpFuncAttrs();
  if (!op_func_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = mlir::cast<mlir::ArrayAttr>(attr).getValue();
      assert(key_value.size() == 2 && "invalid named attribute format.");
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
void PrintExecuteOpImpl(OpAsmPrinter &p, OpTy op) {
  auto op_attrs = op.getOpAttrs();
  if (!op_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = mlir::cast<ArrayAttr>(attr).getValue();
      assert(key_value.size() == 2 && "invalid named attribute format.");
      auto key = key_value[0];
      auto value = key_value[1];

      p << mlir::cast<StringAttr>(key).getValue();
      p << " = ";
      p << value;
    };

    auto op_attr_array = op_attrs.getValue();
    p << " {";
    interleaveComma(op_attr_array, p, print_key_value);
    p << '}';
  }
}

ParseResult ParseExecuteOpImpl(OpAsmParser &parser, OperationState &result,
                               int num_chains, bool has_func_attr = false);

}  // namespace corert
}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OPDEFS_CORERT_UTILS_H_
