/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/quantization/common/func.h"
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/core/platform/test.h"

namespace mlir::quant {

using ::testing::Test;

class QuantizationTestBase : public Test {
 protected:
  QuantizationTestBase()
      : ctx_(stablehlo::CreateMlirContextForQuantization()),
        builder_(ctx_.get()) {
    ctx_->loadDialect<arith::ArithDialect, mlir::stablehlo::StablehloDialect,
                      func::FuncDialect, TF::TensorFlowDialect,
                      tf_saved_model::TensorFlowSavedModelDialect,
                      tf_executor::TensorFlowExecutorDialect,
                      quant::QuantDialect, ir::TFQuantDialect>();
  }

  // Parses `module_op_str` to create a `ModuleOp`.
  OwningOpRef<ModuleOp> ParseModuleOpString(
      const absl::string_view module_op_str) {
    return parseSourceString<ModuleOp>(module_op_str, ctx_.get());
  }

  // Convenience function that returns the first operation of type `OpT` from
  // the `@main` function in `module_op`. Useful when testing with a text
  // representation of a `ModuleOp` containing a single function `@main`.
  // Returns `failure` iff there is no `@main` or no such operation is found in
  // `@main`.
  template <typename OpT>
  FailureOr<OpT> FindFirstOpFromMainFunc(ModuleOp module_op) {
    func::FuncOp main_func_op = FindMainFuncOp(module_op);
    if (main_func_op == nullptr) return failure();

    auto ops = main_func_op.getOps<OpT>();
    if (ops.empty()) return failure();

    return *ops.begin();
  }

  std::unique_ptr<MLIRContext> ctx_;
  OpBuilder builder_;
};

}  // namespace mlir::quant

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_
