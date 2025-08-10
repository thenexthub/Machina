/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace tf2xla {
namespace internal {

namespace {

using mlir::BlockArgument;
using mlir::failure;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OpOperand;
using mlir::StringAttr;
using mlir::success;
using mlir::Value;
using mlir::WalkResult;
using mlir::func::FuncOp;
using mlir::TF::ReadVariableOp;
using mlir::tf_device::ReplicateOp;

#define GEN_PASS_DEF_HOISTBROADCASTREADPASS
#include "machina/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

constexpr char kFuncDeviceAttr[] = "tf.device";
constexpr char kCpuDeviceType[] = "CPU";

struct HoistBroadcastRead
    : public impl::HoistBroadcastReadPassBase<HoistBroadcastRead> {
  void runOnOperation() override;
};

// Get the ancestor of `descendant` that is a direct child of `ancestor`.
Operation* GetAncestorBelow(Operation* descendant, Operation* ancestor) {
  Operation* parent = descendant->getParentOp();
  if (!parent) return nullptr;
  if (parent == ancestor) return descendant;
  return GetAncestorBelow(parent, ancestor);
}

// `is_cpu_read` is set to `true` iff `read` is on a resource with device type
// CPU.
LogicalResult IsCpuRead(FuncOp func, ReadVariableOp read, bool& is_cpu_read) {
  if (auto arg = mlir::dyn_cast<BlockArgument>(read->getOperand(0))) {
    if (arg.getOwner() != &(func.front())) {
      is_cpu_read = false;
      return success();
    }
    if (auto attr = func.getArgAttrOfType<StringAttr>(arg.getArgNumber(),
                                                      kFuncDeviceAttr)) {
      std::string device = attr.getValue().str();
      machina::DeviceNameUtils::ParsedName parsed_name;
      if (!machina::DeviceNameUtils::ParseFullName(device, &parsed_name)) {
        return read->emitOpError() << "invalid device '" << device << "'";
      }
      is_cpu_read = parsed_name.type == kCpuDeviceType;
      return success();
    }
  }
  is_cpu_read = false;
  return success();
}

// Get the reads to hoist in the `replicate`.
LogicalResult GetReads(FuncOp func, ReplicateOp replicate,
                       toolchain::SmallVector<ReadVariableOp, 4>& reads) {
  for (Operation& op : replicate.getBody().front()) {
    if (auto read = toolchain::dyn_cast<ReadVariableOp>(&op)) {
      bool is_cpu_read;
      if (failed(IsCpuRead(func, read, is_cpu_read))) return failure();
      if (is_cpu_read) reads.push_back(read);
    }
  }
  return success();
}

// Move reads above the `replicate`. Skip reads that come after a write to the
// same resource.
void MoveReads(ReplicateOp replicate,
               toolchain::SmallVector<ReadVariableOp, 4>& reads) {
  for (ReadVariableOp read : reads) {
    Value res = read.getResource();
    Operation* scope = res.getParentBlock()->getParentOp();
    if (!scope->isProperAncestor(replicate)) continue;
    bool has_conflicting_write = false;
    for (OpOperand& use : res.getUses()) {
      Operation* using_op = use.getOwner();
      if (using_op == read) continue;
      if (!replicate->isProperAncestor(using_op)) continue;
      Operation* peer = GetAncestorBelow(using_op, replicate);
      if (read->isBeforeInBlock(peer)) continue;
      if (toolchain::isa<ReadVariableOp>(peer)) continue;
      has_conflicting_write = true;
    }
    if (has_conflicting_write) continue;
    read->moveBefore(replicate);
  }
}

// Hoist `ReadVariableOp`s above the `tf_device.replicate`s.
void HoistBroadcastRead::runOnOperation() {
  FuncOp func = getOperation();

  auto result = func.walk([&](ReplicateOp replicate) {
    toolchain::SmallVector<ReadVariableOp, 4> reads;
    if (failed(GetReads(func, replicate, reads)))
      return WalkResult::interrupt();
    MoveReads(replicate, reads);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateHoistBroadcastReadPass() {
  return std::make_unique<HoistBroadcastRead>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
