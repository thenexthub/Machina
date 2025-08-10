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

#include "machina/core/transforms/drop_unregistered_attribute/pass.h"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_DROPOUTPUTSHAPESATTR
#include "machina/core/transforms/passes.h.inc"

struct DropOutputShapesAttrPass
    : impl::DropOutputShapesAttrBase<DropOutputShapesAttrPass> {
  LogicalResult initialize(MLIRContext* context) override {
    for (auto& str : skip_) {
      skip_id.insert(StringAttr::get(context, str));
    }
    return success();
  }
  void runOnOperation() override {
    Operation* op = getOperation();
    op->walk([this](Operation* op) {
      if (!skip_id.count(op->getName().getIdentifier()))
        op->removeAttr("_output_shapes");
    });
  }

  // Set of operation types to skip over.
  DenseSet<StringAttr> skip_id;
};

}  // namespace

std::unique_ptr<Pass> CreateDropOutputShapesAttrPass() {
  return std::make_unique<DropOutputShapesAttrPass>();
}

}  // namespace tfg
}  // namespace mlir
