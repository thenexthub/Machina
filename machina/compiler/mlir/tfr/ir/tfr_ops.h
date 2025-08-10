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

#ifndef MACHINA_COMPILER_MLIR_TFR_IR_TFR_OPS_H_
#define MACHINA_COMPILER_MLIR_TFR_IR_TFR_OPS_H_

#include "toolchain/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/DialectImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/FunctionInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace TFR {

constexpr char kAttrArgumentNameAttr[] = "tfr.name";
constexpr char kAttrArgumentDefaultAttr[] = "tfr.default";
constexpr char kAttrArgumentTypeAttr[] = "tfr.type";

class TFRDialect : public Dialect {
 public:
  explicit TFRDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tfr"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type ty, DialectAsmPrinter &os) const override;
};

}  // namespace TFR
}  // namespace mlir

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfr/ir/tfr_ops.h.inc"

#endif  // MACHINA_COMPILER_MLIR_TFR_IR_TFR_OPS_H_
