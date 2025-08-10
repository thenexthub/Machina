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

#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/mark_initialized_variables.h"
#include "machina/compiler/mlir/machina/utils/fake_session.h"

namespace mlir {
namespace {

// This pass is only available in the tf-opt binary for testing.
class MarkInitializedVariablesTestPass
    : public PassWrapper<MarkInitializedVariablesTestPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkInitializedVariablesTestPass)

  StringRef getArgument() const final {
    return "tf-saved-model-mark-initialized-variables-test";
  }

  StringRef getDescription() const final {
    return "Mark variables as initialized or not.";
  }

  void runOnOperation() override {
    TF::test_util::FakeSession session;
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            getOperation(), &session)))
      return signalPassFailure();
  }
};

// This pass is only available in the tf-opt binary for testing.
class MarkInitializedVariablesInvalidSessionTestPass
    : public PassWrapper<MarkInitializedVariablesInvalidSessionTestPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MarkInitializedVariablesInvalidSessionTestPass)

  StringRef getArgument() const final {
    return "tf-saved-model-mark-initialized-variables-invalid-session-test";
  }

  StringRef getDescription() const final {
    return "Mark variables as initialized or not, but with invalid session.";
  }

  void runOnOperation() override {
    // Pass an invalid session argument, which is a nullptr.
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            getOperation(), /*session=*/nullptr)))
      return signalPassFailure();
  }
};

}  // namespace

namespace tf_saved_model {

static PassRegistration<MarkInitializedVariablesTestPass>
    mark_initialized_variables_test_pass;

static PassRegistration<MarkInitializedVariablesInvalidSessionTestPass>
    mark_initialized_variables_invalid_session_test_pass;

}  // namespace tf_saved_model
}  // namespace mlir
