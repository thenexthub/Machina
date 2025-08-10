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

// This transformation pass converts unified compilation and replication
// attributes into legacy attributes. For example,  _replication_info=X
// and _xla_compile_device_type=TPU should be replaced with _tpu_replicate=X.
// This ensures the unified attributes not get exposed outside of the MLIR
// bridge with V1 pipeline in some cases.

#include <memory>

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Debug.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_CONVERTTOLEGACYCOMPILEANDREPLICATEATTRIBUTESPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct ConvertToLegacyCompileAndReplicateAttributesPass
    : public impl::ConvertToLegacyCompileAndReplicateAttributesPassBase<
          ConvertToLegacyCompileAndReplicateAttributesPass> {
  void runOnOperation() override;
};

LogicalResult ConvertToLegacyAttributes(func::FuncOp func_op) {
  auto result = func_op->walk([&](mlir::Operation* op) {
    if (failed(TF::HasValidCompilationAndReplicationAttributes(*op)))
      return WalkResult::interrupt();
    if (op->hasAttr(TF::kReplicationInfoAttr)) {
      op->setAttr(TF::kTpuReplicateAttr, op->getAttr(TF::kReplicationInfoAttr));
      op->removeAttr(TF::kReplicationInfoAttr);
      op->removeAttr(TF::kCompileDeviceTypeAttr);
    }
    return mlir::WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void ConvertToLegacyCompileAndReplicateAttributesPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  if (failed(ConvertToLegacyAttributes(func_op))) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertToLegacyCompileAndReplicateAttributesPass() {
  return std::make_unique<ConvertToLegacyCompileAndReplicateAttributesPass>();
}

}  // namespace TFTPU
}  // namespace mlir
