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

#include <memory>

#include "machina/compiler/mlir/machina/transforms/initialize_variables_in_session_init.h"
#include "machina/compiler/mlir/machina/transforms/test_passes.h"
#include "machina/compiler/mlir/machina/utils/fake_session.h"

namespace mlir {
namespace tf_test {
namespace {

#define GEN_PASS_DEF_INITIALIZEVARIABLESINSESSIONINITIALIZERPASS
#include "machina/compiler/mlir/machina/transforms/test_passes.h.inc"

class InitializeVariablesInSessionInitializerPass
    : public impl::InitializeVariablesInSessionInitializerPassBase<
          InitializeVariablesInSessionInitializerPass> {
 public:
  void runOnOperation() final {
    static machina::Session* session = new TF::test_util::FakeSession();
    if (failed(tf_saved_model::InitializeVariablesInSessionInitializer(
            getOperation(), session)))
      signalPassFailure();
  }

 private:
};
}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateInitializeVariablesInSessionInitializerTestPass() {
  return std::make_unique<InitializeVariablesInSessionInitializerPass>();
}

}  // namespace tf_test
}  // namespace mlir
