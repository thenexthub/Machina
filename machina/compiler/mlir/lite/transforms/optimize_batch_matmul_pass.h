/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_BATCH_MATMUL_PASS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_BATCH_MATMUL_PASS_H_

#include "toolchain/ADT/APFloat.h"
#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/pass.h"
#include "machina/compiler/mlir/lite/transforms/pass_options.h"

namespace mlir {
namespace TFL {

// Optimize FC with BatchMatmul within the TensorFlow Lite dialect.

class OptimizeBatchMatmulPass
    : public TFL::Pass<OptimizeBatchMatmulPass, EmptyPassOptions,
                       func::FuncOp> {
 public:
  OptimizeBatchMatmulPass() = default;
  OptimizeBatchMatmulPass(const OptimizeBatchMatmulPass &other) {}

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static toolchain::StringRef GetArgument() { return "tfl-optimize-batch-matmul"; }

  static toolchain::StringRef GetDescription() {
    return "Optimize FC with BatchMatmul within the TensorFlow Lite dialect.";
  }

  /// Returns the derived pass name.
  static toolchain::StringRef GetName() { return "OptimizeBatchMatmulPass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeBatchMatmulPass)
};

}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_BATCH_MATMUL_PASS_H_
