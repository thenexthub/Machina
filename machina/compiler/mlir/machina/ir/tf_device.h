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

// This file defines the tf_device dialect: it contains operations that model
// TensorFlow's actions to launch computations on accelerator devices.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_DEVICE_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_DEVICE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain

namespace mlir {
namespace tf_device {

// The TensorFlow Device dialect.
//
// This dialect contains operations to describe/launch computations on devices.
// These operations do not map 1-1 to TensorFlow ops and requires a lowering
// pass later to transform them into Compile/Run op pairs, like XlaCompile and
// XlaRun.
class TensorFlowDeviceDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_device"; }
  // Constructing TensorFlowDevice dialect under an non-null MLIRContext.
  explicit TensorFlowDeviceDialect(MLIRContext* context);
};

}  // namespace tf_device
}  // namespace mlir

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "machina/compiler/mlir/machina/ir/tf_device.h.inc"

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_DEVICE_H_
