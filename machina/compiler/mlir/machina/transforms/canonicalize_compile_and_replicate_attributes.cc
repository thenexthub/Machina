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

// This transformation pass converts existing compilation and replication
// attributes into unified attributes. For example, A _tpu_replicate=X
// should be replaced with _xla_compile_device_type=TPU and
// _replication_info=X attributes by the conversion. An _XlaMustCompile=true
// should be replaced with _xla_compile_device_type with the value of device
// attribute.

#include <memory>

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Debug.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"
#include "machina/core/util/device_name_utils.h"

#define DEBUG_TYPE "tf-canonicalize-compile-and-replicate-attributes"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_CANONICALIZECOMPILEANDREPLICATEATTRIBUTESPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct CanonicalizeCompileAndReplicateAttributesPass
    : public impl::CanonicalizeCompileAndReplicateAttributesPassBase<
          CanonicalizeCompileAndReplicateAttributesPass> {
  void runOnOperation() override;
};

void CanonicalizeCompileAndReplicateAttributesPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  ModuleOp module_op = func_op->getParentOfType<ModuleOp>();
  mlir::OpBuilder builder(module_op.getContext());

  auto walk_result = func_op->walk([&](mlir::Operation* op) {
    // Convert `_tpu_replicate`.
    if (op->hasAttr(TF::kTpuReplicateAttr)) {
      op->setAttr(machina::kReplicationInfoAttr,
                  op->getAttr(TF::kTpuReplicateAttr));
      op->removeAttr(machina::kTpuReplicateAttr);
      op->setAttr(machina::kCompileDeviceTypeAttr,
                  builder.getStringAttr(machina::kTpuDevice));
    }

    // Convert `_XlaMustCompile`.
    if (op->hasAttr(machina::kMustCompileAttr)) {
      bool must_compile_attr_val =
          op->getAttrOfType<BoolAttr>(machina::kMustCompileAttr).getValue();
      op->removeAttr(machina::kMustCompileAttr);
      if (!must_compile_attr_val) {
        if (op->hasAttr(machina::kCompileDeviceTypeAttr)) {
          op->emitOpError()
              << "has both '" << machina::kMustCompileAttr
              << " = false' and '" << machina::kCompileDeviceTypeAttr
              << "' attribute which contradicts each other";
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }
      if (op->hasAttr(machina::kCompileDeviceTypeAttr)) {
        return mlir::WalkResult::advance();
      }
      auto device_attr = op->getAttrOfType<StringAttr>(machina::kDeviceAttr);
      if (!device_attr) {
        op->setAttr(machina::kCompileDeviceTypeAttr,
                    builder.getStringAttr(machina::kEmptyDevice));
        return mlir::WalkResult::advance();
      }
      machina::DeviceNameUtils::ParsedName parsed_name;
      machina::DeviceNameUtils::ParseFullOrLocalName(device_attr.getValue(),
                                                        &parsed_name);
      auto device_type = builder.getStringAttr(parsed_name.type);
      if (failed(IsValidDeviceTypeOrEmpty(device_type))) {
        op->emitOpError() << "'" << machina::kDeviceAttr << "'"
                          << " has invalid value";
        return mlir::WalkResult::interrupt();
      }
      op->setAttr(machina::kCompileDeviceTypeAttr, device_type);
    }

    return mlir::WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateCanonicalizeCompileAndReplicateAttributesPass() {
  return std::make_unique<CanonicalizeCompileAndReplicateAttributesPass>();
}

}  // namespace TF
}  // namespace mlir
