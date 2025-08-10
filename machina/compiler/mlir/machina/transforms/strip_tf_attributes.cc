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

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_STRIPTFATTRIBUTESPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct StripTfAttributesPass
    : public impl::StripTfAttributesPassBase<StripTfAttributesPass> {
  void runOnOperation() override;
};

bool ShouldStripAttr(NamedAttribute &namedAttr) {
  StringRef name = namedAttr.getName().strref();
  if (name.starts_with("tf.") || name.starts_with("tf_")) return true;
  StringRef value = namedAttr.getValue().getDialect().getNamespace();
  return value == "tf" || value.starts_with("tf_");
}

void StripFunction(func::FuncOp func) {
  auto stripAttrs = toolchain::to_vector<4>(toolchain::make_filter_range(
      func->getAttrs(),
      [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
  for (auto namedAttr : stripAttrs) {
    func->removeAttr(namedAttr.getName());
  }

  for (int i = 0; i < func.getNumArguments(); ++i) {
    toolchain::ArrayRef<mlir::NamedAttribute> attrs =
        mlir::function_interface_impl::getArgAttrs(func, i);
    auto stripAttrs = toolchain::to_vector<4>(toolchain::make_filter_range(
        attrs,
        [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      func.removeArgAttr(i, namedAttr.getName());
    }
  }

  for (int i = 0; i < func.getNumResults(); ++i) {
    auto stripAttrs = toolchain::to_vector<4>(toolchain::make_filter_range(
        func.getResultAttrs(i),
        [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      func.removeResultAttr(i, namedAttr.getName());
    }
  }
}

void StripTfAttributesPass::runOnOperation() {
  ModuleOp module = getOperation();

  // strip module itself
  auto stripAttrs = toolchain::to_vector<4>(toolchain::make_filter_range(
      module->getAttrs(),
      [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
  for (auto namedAttr : stripAttrs) {
    module->removeAttr(namedAttr.getName());
  }

  // strip functions in module
  module.walk([&](func::FuncOp func) { StripFunction(func); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateStripTfAttributesPass() {
  return std::make_unique<StripTfAttributesPass>();
}

}  // namespace TF
}  // namespace mlir
