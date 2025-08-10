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
#ifndef MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_MLRT_DIALECT_H_
#define MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_MLRT_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain

namespace mlrt {
namespace compiler {

class MlrtDialect : public mlir::Dialect {
 public:
  explicit MlrtDialect(mlir::MLIRContext *context);
  static toolchain::StringRef getDialectNamespace() { return "mlrt"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;
};

// The MLIR type represents a C++ mlrt::Future.
class FutureType
    : public mlir::Type::TypeBase<FutureType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
  static constexpr mlir::StringLiteral name = "mlrt.compiler.future";
};

// The MLIR type represents a C++ mlrt::Promise.
class PromiseType
    : public mlir::Type::TypeBase<PromiseType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
  static constexpr mlir::StringLiteral name = "mlrt.compiler.promise";
};

// The MLIR type represents a C++ mlrt::AsyncHandle.
class AsyncHandleType : public mlir::Type::TypeBase<AsyncHandleType, mlir::Type,
                                                    mlir::TypeStorage> {
 public:
  using Base::Base;
  static constexpr mlir::StringLiteral name = "mlrt.compiler.async_handle";
};

}  // namespace compiler
}  // namespace mlrt

#endif  // MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_MLRT_DIALECT_H_
