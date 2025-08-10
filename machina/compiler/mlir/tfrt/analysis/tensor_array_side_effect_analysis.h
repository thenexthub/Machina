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
#ifndef MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_TENSOR_ARRAY_SIDE_EFFECT_ANALYSIS_H_
#define MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_TENSOR_ARRAY_SIDE_EFFECT_ANALYSIS_H_

#include "toolchain/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain

namespace machina {
namespace tfrt_compiler {

// Return true if it is a TensorArrayOp, eg. TensorArrayV3Op.
bool IsTensorArrayOp(mlir::Operation* op);

// This class provides utilities for analyzing side effects for TensorArray ops
// in the graph. mlir::TF::SideEffectAnalysis currently produces suboptimal
// side-effect analysis for TensorArray ops. On the other hand, control
// dependencies are already sorted out for TensorArray ops in the original TF
// graph. Each TensorArray op will take or produce a `flow` value and they are
// already properly chained in the origninal TF graph.
class TensorArraySideEffectAnalysis {
 public:
  explicit TensorArraySideEffectAnalysis(mlir::ModuleOp module);

  // Return if the function contains only non-side-effecting ops or TensorArray
  // ops.
  bool HasAtMostTensorArrayEffect(mlir::func::FuncOp func_op) const {
    return set_.count(func_op) > 0;
  }

 private:
  toolchain::DenseSet<mlir::func::FuncOp> set_;
};

}  // namespace tfrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_ANALYSIS_TENSOR_ARRAY_SIDE_EFFECT_ANALYSIS_H_
