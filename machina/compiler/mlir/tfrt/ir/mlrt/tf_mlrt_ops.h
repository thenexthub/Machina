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
#ifndef MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_
#define MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_side_effects.h"
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"  // from @tf_runtime
#include "tfrt/compiler/opdefs/tfrt_traits.h"  // from @tf_runtime

namespace machina {
namespace tf_mlrt {

class TensorflowMlrtDialect : public mlir::Dialect {
 public:
  explicit TensorflowMlrtDialect(mlir::MLIRContext *context);
  static toolchain::StringRef getDialectNamespace() { return "tf_mlrt"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;
};

// The MLIR type represents a machina::Tensor.
class TFTensorType
    : public mlir::Type::TypeBase<TFTensorType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
  static constexpr mlir::StringLiteral name = "machina.tf_mlrt.tf_tensor";
};

// The MLIR type represents a machina::Device*
class TFDeviceType
    : public mlir::Type::TypeBase<TFDeviceType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
  static constexpr mlir::StringLiteral name = "machina.tf_mlirt.tf_device";
};

}  // namespace tf_mlrt
}  // namespace machina

#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h.inc"
#define GET_OP_CLASSES
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_ops.h.inc"

#endif  // MACHINA_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_
