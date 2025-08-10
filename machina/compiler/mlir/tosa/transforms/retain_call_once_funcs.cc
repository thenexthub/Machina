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

#include <memory>

#include "toolchain/ADT/StringExtras.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "retain-call-once-funcs"
#define DEBUG_TYPE PASS_NAME

namespace mlir::tosa {

#define GEN_PASS_DEF_RETAINCALLONCEFUNCS
#include "machina/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

class RetainCallOnceFuncsPass
    : public impl::RetainCallOnceFuncsBase<RetainCallOnceFuncsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    toolchain::DenseMap<StringRef, func::FuncOp> funcMap;
    for (auto func : moduleOp.getOps<mlir::func::FuncOp>()) {
      funcMap[func.getSymName()] = func;
    }

    for (auto func : moduleOp.getOps<mlir::func::FuncOp>()) {
      for (auto callOnce : func.getOps<mlir::TFL::CallOnceOp>()) {
        auto callFunc = funcMap[callOnce.getSessionInitFunction()];
        callOnce->setAttr("session_init_function_symbol",
                          SymbolRefAttr::get(callFunc));
      }
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createRetainCallOnceFuncsPass() {
  return std::make_unique<RetainCallOnceFuncsPass>();
}

}  // namespace mlir::tosa
