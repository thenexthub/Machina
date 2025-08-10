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
#include "machina/compiler/mlir/tfrt/analysis/tensor_array_side_effect_analysis.h"

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace machina {
namespace tfrt_compiler {

bool IsTensorArrayOp(mlir::Operation* op) {
  return toolchain::isa<mlir::TF::TensorArrayV3Op, mlir::TF::TensorArrayScatterV3Op,
                   mlir::TF::TensorArrayGatherV3Op,
                   mlir::TF::TensorArrayReadV3Op,
                   mlir::TF::TensorArrayWriteV3Op>(op);
}

static bool FunctionContainsOnlyNoSideEffectOpOrTensorArrayOp(
    mlir::func::FuncOp func_op) {
  for (mlir::Operation& op : func_op.front()) {
    if (!mlir::isMemoryEffectFree(&op) && !IsTensorArrayOp(&op)) return false;
  }

  return true;
}

TensorArraySideEffectAnalysis::TensorArraySideEffectAnalysis(
    mlir::ModuleOp module) {
  for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
    if (FunctionContainsOnlyNoSideEffectOpOrTensorArrayOp(func_op)) {
      set_.insert(func_op);
    }
  }
}

}  // namespace tfrt_compiler
}  // namespace machina
