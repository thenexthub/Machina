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

#include <iterator>
#include <memory>
#include <utility>

#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"
#include "machina/xla/hlo/builder/sharding_builder.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "machina/dtensor/mlir/dtensor_mlir_passes.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/op_utils.h"
#include "machina/dtensor/mlir/spmd_expander_common.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORTPUINTEGRATION
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

// Adds metadata used in TPU Compilation to `cluster` as attributes.
void AddMetadataToTPUCluster(const Mesh& mesh_config,
                             mlir::tf_device::ClusterOp cluster,
                             mlir::OpBuilder* builder) {
  cluster->setAttr("_tpu_replicate",
                   builder->getStringAttr(mesh_config.ToString()));
  cluster->setAttr("step_marker_location", builder->getStringAttr(""));
  cluster->setAttr("padding_map", builder->getArrayAttr({}));
  cluster->setAttr("use_spmd_for_xla_partitioning",
                   builder->getBoolAttr(false));
  cluster->setAttr(machina::kTopologyAttr, builder->getStringAttr(""));
  cluster->setAttr(machina::kDeviceAssignmentAttr,
                   builder->getArrayAttr({}));
  cluster->setAttr(machina::kNumCoresPerReplicaAttr,
                   builder->getI64IntegerAttr(1));
}

// TODO(hongjunchoi): Implement cluster inlining pass so that there are no
// nested tf_device.cluster ops with same mesh.
void IdentifyTPUFunctions(
    mlir::ModuleOp module, toolchain::SmallVectorImpl<Mesh>* tpu_meshs,
    toolchain::SmallVectorImpl<mlir::TF::StatefulPartitionedCallOp>* tpu_functions) {
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) return;

  for (auto call : main_func.getOps<mlir::TF::StatefulPartitionedCallOp>()) {
    auto mesh_or_status = Mesh::FromString(string(call.getConfig()));
    // Function calls created by end users instead of being converted from
    // tf_device.cluster do not have a serialized mesh as a config attribute. We
    // ignore the error returned from parsing in this case.
    if (!mesh_or_status.ok()) return;
    bool skip_xla_compilation = false;
    if (call->hasAttr(kSkipXlaCompilation)) {
      skip_xla_compilation =
          call->getAttrOfType<mlir::BoolAttr>(kSkipXlaCompilation).getValue();
    }
    if (mesh_or_status->is_tpu_mesh() && !skip_xla_compilation) {
      tpu_functions->emplace_back(call);
      tpu_meshs->emplace_back(std::move(mesh_or_status.value()));
    }
  }
}

mlir::LogicalResult CreateTPUCluster(
    mlir::TF::StatefulPartitionedCallOp tpu_call, mlir::OpBuilder* builder,
    mlir::tf_device::ClusterOp* newly_created_cluster) {
  auto function = MaybeFindFunction(tpu_call);
  if (!function)
    return tpu_call.emitOpError(
        "failed during TPU Integration as Func op TPU mesh was not found");

  auto& function_block = function->getCallableRegion()->front();
  builder->setInsertionPointToStart(&function_block);

  auto cluster = builder->create<mlir::tf_device::ClusterOp>(
      tpu_call.getLoc(), function->getResultTypes());
  cluster.getBody().push_back(new mlir::Block);

  auto& function_body = function_block.getOperations();
  cluster.GetBody().getOperations().splice(
      cluster.GetBody().getOperations().begin(), function_body,
      std::next(function_body.begin()), std::prev(function_body.end()));

  builder->setInsertionPointToEnd(&cluster.GetBody());
  mlir::Operation* function_block_terminator = function_block.getTerminator();
  builder->create<mlir::tf_device::ReturnOp>(
      tpu_call.getLoc(), function_block_terminator->getOperands());

  function_block_terminator->setOperands(cluster.getResults());

  *newly_created_cluster = cluster;
  return mlir::success();
}

struct DTensorTPUIntegration
    : public impl::DTensorTPUIntegrationBase<DTensorTPUIntegration> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
    registry.insert<mlir::tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    auto module = getOperation();
    toolchain::SmallVector<mlir::TF::StatefulPartitionedCallOp, 4> tpu_functions;
    toolchain::SmallVector<Mesh, 4> tpu_meshes;
    IdentifyTPUFunctions(module, &tpu_meshes, &tpu_functions);

    for (auto tpu_function_and_mesh : toolchain::zip(tpu_meshes, tpu_functions)) {
      mlir::tf_device::ClusterOp cluster;

      if (mlir::failed(CreateTPUCluster(std::get<1>(tpu_function_and_mesh),
                                        &op_builder, &cluster)))
        return signalPassFailure();

      AddMetadataToTPUCluster(std::get<0>(tpu_function_and_mesh), cluster,
                              &op_builder);
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorTPUIntegration() {
  return std::make_unique<DTensorTPUIntegration>();
}

}  // namespace dtensor
}  // namespace machina
