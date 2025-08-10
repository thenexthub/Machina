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
#include <string>

#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"
#include "machina/dtensor/mlir/layout_parsing.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSOROPTODEVICECLUSTER
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

// Extracts mesh config from the Op.
// We currently hard extract mesh information from all the args and assume they
// are the same. This should not be the case when we have multiple functions.
mlir::LogicalResult WrapDeviceCluster(mlir::OpBuilder *builder,
                                      mlir::Operation *op) {
  // Create new tf_device.cluster op wrapping a single operation.
  builder->setInsertionPoint(op);
  auto cluster = builder->create<mlir::tf_device::ClusterOp>(
      op->getLoc(), op->getResultTypes());
  if (auto layout_op = toolchain::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(
                                    layout_op.getLayout().mesh().ToString()));
  } else if (auto copy_to_mesh = toolchain::dyn_cast<mlir::TF::RelayoutOp>(op)) {
    const std::string layout_string = copy_to_mesh.getLayout().str();
    auto layout = Layout::FromString(layout_string);
    if (!layout.ok())
      return op->emitOpError(toolchain::formatv(
          "Found tf.Relayout Op with unparsable layout: {0}", layout_string));

    cluster->setAttr(kMeshAttr,
                     builder->getStringAttr(layout->mesh().ToString()));
  } else {
    // If mesh configuration can be inferred from the op directly, use the mesh
    // information from op attribute directly. If op is not annotated with mesh
    // information, then mesh will be inferred in following
    // DTensorMeshPropagation pass and will be inferred from consumers or
    // operands.
    auto status_or_mesh = ExtractDeviceMeshFromOp(op);

    if (!status_or_mesh.ok())
      return op->emitOpError(
          toolchain::formatv("failed to wrap to device cluster. {0}",
                        status_or_mesh.status().message()));

    const auto mesh_config = status_or_mesh.value();
    if (mesh_config)
      cluster->setAttr(kMeshAttr,
                       builder->getStringAttr(mesh_config->ToString()));
  }

  op->replaceAllUsesWith(cluster);

  cluster.getBody().push_back(new mlir::Block);

  builder->setInsertionPointToEnd(&cluster.GetBody());
  builder->create<mlir::tf_device::ReturnOp>(op->getLoc(), op->getResults());

  // Move `op` inside newly created `ClusterOp`.
  op->moveBefore(cluster.GetBody().getTerminator());

  return mlir::success();
}

// MLIR pass that wraps tf_device.cluster op to every TF op.
struct DTensorOpToDeviceClusterPass
    : public impl::DTensorOpToDeviceClusterBase<DTensorOpToDeviceClusterPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
    registry.insert<mlir::tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::OpBuilder op_builder(&context);
    mlir::Dialect *tf =
        getContext().getLoadedDialect<mlir::TF::TensorFlowDialect>();

    auto walk_result = getOperation().walk([&](mlir::Operation *operation) {
      const auto op_dialect = operation->getDialect();
      // Only TF dialects are supported for layout propagation.
      if (op_dialect != tf) return mlir::WalkResult::advance();

      // For control flow operations, tf.yield ops exists and should not be
      // wrapped to tf_device.cluster as the op does not need to be transformed
      // in SPMD expansion and tf.If/tf.While op require all ops to terminate
      // with tf.Yield op. Wrapping yield op in tf_device.cluster invalidates
      // this invariant.
      if (toolchain::isa<mlir::TF::YieldOp>(operation))
        return mlir::WalkResult::advance();

      if (mlir::failed(WrapDeviceCluster(&op_builder, operation)))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });

    if (walk_result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorOpToDeviceClusterPass() {
  return std::make_unique<DTensorOpToDeviceClusterPass>();
}

}  // namespace dtensor
}  // namespace machina
