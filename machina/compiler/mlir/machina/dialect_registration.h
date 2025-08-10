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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_DIALECT_REGISTRATION_H_
#define MACHINA_COMPILER_MLIR_MACHINA_DIALECT_REGISTRATION_H_

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // part of Codira Toolchain
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/core/ir/ops.h"

namespace mlir {
// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
// If include_extensions is set (default), also registers extensions. Otherwise
// it is the responsibility of the caller, typically required when the registry
// is appended to the context in a parallel context, which does not allow for
// extensions to be added.
inline void RegisterAllTensorFlowDialectsImpl(DialectRegistry &registry,
                                              bool include_extensions = true) {
  registry
      .insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
              mlir::ml_program::MLProgramDialect, mlir::TF::TensorFlowDialect,
              mlir::tf_type::TFTypeDialect, mlir::cf::ControlFlowDialect,
              mlir::tf_device::TensorFlowDeviceDialect,
              mlir::tf_executor::TensorFlowExecutorDialect,
              mlir::tf_saved_model::TensorFlowSavedModelDialect,
              mlir::tfg::TFGraphDialect>();
  if (include_extensions) {
    mlir::func::registerAllExtensions(registry);
  }
}

// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
inline void RegisterAllTensorFlowDialects(DialectRegistry &registry) {
  RegisterAllTensorFlowDialectsImpl(registry, true);
}
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_DIALECT_REGISTRATION_H_
