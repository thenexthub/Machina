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

#include <memory>

#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/lift_variables.h"
#include "machina/compiler/mlir/machina/transforms/test_passes.h"
#include "machina/compiler/mlir/machina/utils/fake_session.h"

namespace mlir {
namespace tf_test {
namespace {
using ::machina::Session;

#define GEN_PASS_DEF_LIFTVARIABLESTESTPASS
#include "machina/compiler/mlir/machina/transforms/test_passes.h.inc"

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesTestPass
    : public impl::LiftVariablesTestPassBase<LiftVariablesTestPass> {
 public:
  LiftVariablesTestPass() { session_ = new TF::test_util::FakeSession(); }

  ~LiftVariablesTestPass() override { delete session_; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(tf_saved_model::LiftVariables(
            module, session_, /*import_variables_as_dense_resources=*/
            import_variables_as_dense_resources_)))
      signalPassFailure();
  }

 private:
  Session* session_;
};

#define GEN_PASS_DEF_LIFTVARIABLESINVALIDSESSIONTESTPASS
#include "machina/compiler/mlir/machina/transforms/test_passes.h.inc"

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesInvalidSessionTestPass
    : public impl::LiftVariablesInvalidSessionTestPassBase<
          LiftVariablesInvalidSessionTestPass> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Pass an invalid session argument, which is a nullptr.
    if (failed(tf_saved_model::LiftVariables(module, /*session=*/nullptr)))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace tf_test

namespace tf_test {

std::unique_ptr<OperationPass<ModuleOp>> CreateLiftVariablesTestPass() {
  return std::make_unique<tf_test::LiftVariablesTestPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftVariablesInvalidSessionTestPass() {
  return std::make_unique<tf_test::LiftVariablesInvalidSessionTestPass>();
}

}  // namespace tf_test
}  // namespace mlir
