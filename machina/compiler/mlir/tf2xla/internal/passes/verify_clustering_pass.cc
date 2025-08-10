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
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"
#include "machina/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"

namespace machina {
namespace tf2xla {
namespace internal {

namespace {

#define GEN_PASS_DEF_VERIFYCLUSTERINGPASS
#include "machina/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

using mlir::Operation;
using mlir::WalkResult;

class VerifyClusteringPass
    : public impl::VerifyClusteringPassBase<VerifyClusteringPass> {
 public:
  void runOnOperation() override;
};

void VerifyClusteringPass::runOnOperation() {
  Operation* func_op = getOperation();

  auto walk_result = func_op->walk([&](Operation* op) {
    if (!machina::tf2xla::internal::IsInBridgeAcceptableDialects(op)) {
      std::string error = "op is in dialect " +
                          op->getDialect()->getNamespace().str() +
                          " not in tf functional dialect";
      op->emitError() << error;
      return WalkResult::interrupt();
    }

    if (op->hasAttr(mlir::TF::kXlaOutsideCompilationAttr)) {
      std::string error =
          "op has outside compilation attribute _xla_outside_compilation which "
          "is not allowed after clustering";
      op->emitError() << error;
      return mlir::WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyClusteringPass() {
  return std::make_unique<VerifyClusteringPass>();
}
}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
