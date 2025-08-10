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

#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_STRIPNOINLINEATTRIBUTEPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// This tranformation pass strips any "_noinline" attributes from the module.
struct StripNoinlineAttributePass
    : public impl::StripNoinlineAttributePassBase<StripNoinlineAttributePass> {
 public:
  // void runOnOperation() override;
  void runOnOperation() override {
    // Strip the "tf._noinline" attribute from top-level functions.
    for (auto func_op : getOperation().getOps<func::FuncOp>())
      func_op->removeAttr("tf._noinline");
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateStripNoinlineAttributePass() {
  return std::make_unique<StripNoinlineAttributePass>();
}

}  // namespace TF
}  // namespace mlir
