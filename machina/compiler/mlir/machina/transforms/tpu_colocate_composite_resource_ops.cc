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

#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace TFTPU {
namespace {

#define GEN_PASS_DEF_TPUCOLOCATECOMPOSITERESOURCEOPSPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// Pass that co-locates resource ops that use composite device resources
// (packed tensors) with the underlying physical TPU device.
struct TPUColocateCompositeResourceOps
    : public impl::TPUColocateCompositeResourceOpsPassBase<
          TPUColocateCompositeResourceOps> {
  void runOnOperation() override;
};

// Wraps single op in `tf_device.launch` for explicit device assignment.
void WrapOpInLaunch(OpBuilder* builder, Location loc, Operation* op,
                    toolchain::StringRef device) {
  builder->setInsertionPoint(op);
  auto launch = builder->create<tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.getBody().push_back(new Block);
  op->replaceAllUsesWith(launch);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside cluster.
  op->moveBefore(launch.GetBody().getTerminator());
}

toolchain::SmallVector<Operation*, 4> GetResourceOpsUsingCompositeArgsInReplicate(
    tf_device::ReplicateOp replicate) {
  toolchain::SmallVector<Operation*, 4> resource_users;
  const auto add_resource_op_to_list = [&resource_users](Operation* op) {
    if (!toolchain::isa<TF::AssignVariableOp, TF::ReadVariableOp>(op)) return;

    resource_users.emplace_back(op);
  };

  toolchain::SmallVector<Operation*, 4> resource_users_to_visit;
  for (auto composite_arguments : replicate.GetPackedBlockArguments()) {
    for (auto resource_user : composite_arguments.getUsers())
      resource_users_to_visit.emplace_back(resource_user);
  }

  while (!resource_users_to_visit.empty()) {
    toolchain::SmallVector<Operation*, 4> new_resource_users;

    for (auto resource_user : resource_users_to_visit) {
      add_resource_op_to_list(resource_user);

      // Account for pass-through identity ops.
      if (auto pass_through_identity =
              toolchain::dyn_cast<TF::IdentityOp>(resource_user)) {
        for (auto identity_user :
             pass_through_identity.getOutput().getUsers()) {
          new_resource_users.emplace_back(identity_user);
        }
      }
    }
    resource_users_to_visit.swap(new_resource_users);
  }

  return resource_users;
}

void ColocateCompositeResourceOpsInReplicate(
    tf_device::ReplicateOp replicate_op, OpBuilder* builder) {
  auto devices = replicate_op.getDevices();
  if (!devices) return;
  if (!devices.value().get(machina::GetDeviceAliasForLogicalCore(0))) return;

  const auto composite_resource_users =
      GetResourceOpsUsingCompositeArgsInReplicate(replicate_op);
  for (auto resource_user : composite_resource_users) {
    WrapOpInLaunch(builder, resource_user->getLoc(), resource_user,
                   machina::GetDeviceAliasForLogicalCore(0));
  }
}

void TPUColocateCompositeResourceOps::runOnOperation() {
  // Find all the executes first, since we will mutate the nodes around each
  // execute in the same tf_device.replicate op.
  toolchain::SmallVector<tf_device::LaunchOp, 8> execute_launches;
  getOperation().walk([&](tf_device::LaunchOp op) {
    if (op.WrapsSingleOp() &&
        toolchain::isa<TF::TPUExecuteOp, TF::TPUExecuteAndUpdateVariablesOp>(
            op.GetBody().front()))
      execute_launches.push_back(op);
  });

  OpBuilder builder(&getContext());
  for (auto execute_launch : execute_launches) {
    auto replicate = execute_launch->getParentOfType<tf_device::ReplicateOp>();
    if (!replicate) continue;

    ColocateCompositeResourceOpsInReplicate(replicate, &builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUColocateCompositeResourceOps() {
  return std::make_unique<TPUColocateCompositeResourceOps>();
}

}  // namespace TFTPU
}  // namespace mlir
