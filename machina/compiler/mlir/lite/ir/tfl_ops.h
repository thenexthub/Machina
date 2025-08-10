/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This file defines the operations used in the MLIR TensorFlow Lite dialect.

#ifndef MACHINA_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
#define MACHINA_COMPILER_MLIR_LITE_IR_TFL_OPS_H_

#include "mlir/Dialect/Traits.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/DialectImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/TypeSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/LoopLikeInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops_dialect.h.inc"
#include "machina/compiler/mlir/lite/ir/tfl_ops_enums.h.inc"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "machina/compiler/mlir/lite/schema/schema_generated.h"
#include "machina/compiler/mlir/lite/utils/utils.h"
#include "machina/compiler/mlir/machina/ir/tf_traits.h"
#define GET_ATTRDEF_CLASSES
#include "machina/compiler/mlir/lite/ir/tfl_ops_attrdefs.h.inc"

namespace mlir {
namespace TFL {

typedef TFLDialect TensorFlowLiteDialect;

// The Control type is a token-like value that models control dependencies
class ControlType : public Type::TypeBase<ControlType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "tfl.control";
};

#include "machina/compiler/mlir/lite/ir/tfl_ops_interface.h.inc"

}  // end namespace TFL
}  // end namespace mlir

#define GET_OP_CLASSES
#include "machina/compiler/mlir/lite/ir/tfl_ops.h.inc"

#endif  // MACHINA_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
