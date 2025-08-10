/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#include <string>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/StringSet.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"

// Required when using LLVM_DEBUG macro.
#define DEBUG_TYPE "mark-functions-noinline"

namespace mlir {
namespace quant {
namespace {

// Name of the boolean attribute indicating whether the function can be
// inlined or not.
constexpr StringRef kTfNoinlineAttr = "tf._noinline";

// This pass marks functions with the attribute `tf._noinline = true` so that
// they aren't inlined by the `InlinerPass`. The names of the functions to be
// marked noinline should be specified by the `noinline-functions` option.
class MarkFunctionsNoinlinePass
    : public PassWrapper<MarkFunctionsNoinlinePass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkFunctionsNoinlinePass)

  explicit MarkFunctionsNoinlinePass()
      : MarkFunctionsNoinlinePass(
            /*noinline_functions=*/ArrayRef<std::string>{}) {}

  // `noinline_functions` is a list of function names to be marked noinline.
  explicit MarkFunctionsNoinlinePass(
      const ArrayRef<std::string> noinline_functions)
      : noinline_functions_(CreateNoinlineFunctionsOption(noinline_functions)) {
  }

  MarkFunctionsNoinlinePass(const MarkFunctionsNoinlinePass& other)
      : MarkFunctionsNoinlinePass() {
    noinline_functions_ = other.noinline_functions_;
  }

  StringRef getArgument() const final { return "mark-functions-noinline"; }

  StringRef getDescription() const final {
    return "Marks a function whose name is in `noinline-functions` option with "
           "the attribute `tf._noinline = true`. This attributes the function "
           "from being inlined by the `InlinerPass`.";
  }

  void runOnOperation() override;

 private:
  ListOption<std::string> CreateNoinlineFunctionsOption(
      const ArrayRef<std::string> noinline_functions) {
    return {*this, "noinline-functions",
            toolchain::cl::desc(
                "Name of the functions that should be marked "
                "tf._noinline = true to prevent inlining. The name of the "
                "function should exactly match to be marked noinline."),
            toolchain::cl::list_init<std::string>(noinline_functions),
            toolchain::cl::ZeroOrMore};
  }

  // Gets a set of function names from `noinline_functions_`.
  StringSet<> GetNoinlineFunctionsSet() {
    StringSet<> noinline_functions;
    noinline_functions.insert(noinline_functions_.begin(),
                              noinline_functions_.end());
    return noinline_functions;
  }

  // Names of the functions to be marked noinline.
  ListOption<std::string> noinline_functions_;
};

void MarkFunctionsNoinlinePass::runOnOperation() {
  const StringSet<> noinline_functions = GetNoinlineFunctionsSet();

  func::FuncOp func_op = getOperation();
  Builder builder(&getContext());

  // Adds the `tf._noinline = true` attribute to the function if the name
  // matches.
  if (noinline_functions.contains(func_op.getSymName())) {
    func_op->setAttr(kTfNoinlineAttr, builder.getBoolAttr(true));
    LLVM_DEBUG(toolchain::dbgs()
               << "Marked tf._noinline = true: " << func_op.getSymName());
  }
}

static PassRegistration<MarkFunctionsNoinlinePass> pass{};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateMarkFunctionsNoinlinePass(
    const ArrayRef<std::string> noinline_functions) {
  return std::make_unique<MarkFunctionsNoinlinePass>(noinline_functions);
}

}  // namespace quant
}  // namespace mlir
