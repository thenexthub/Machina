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

#include <memory>

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORDESIGNATERESOURCEHANDLEMESH
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

mlir::LogicalResult SetMeshForResourceCreatingCluster(
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder) {
  auto result = cluster.walk([](mlir::Operation* op) {
    if (toolchain::isa<mlir::TF::VarHandleOp, mlir::TF::DestroyResourceOp>(op))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (!result.wasInterrupted()) return mlir::success();

  if (!cluster->hasAttr(kMeshAttr)) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(Mesh::kEmptyMeshString));
  }
  return mlir::success();
}

struct DTensorDesignateResourceHandleMesh
    : public impl::DTensorDesignateResourceHandleMeshBase<
          DTensorDesignateResourceHandleMesh> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto walk_result =
        getOperation().walk([&](mlir::tf_device::ClusterOp cluster) {
          if (mlir::failed(
                  SetMeshForResourceCreatingCluster(cluster, &builder)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorDesignateResourceHandleMesh() {
  return std::make_unique<DTensorDesignateResourceHandleMesh>();
}

}  // namespace dtensor
}  // namespace machina
