/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "absl/log/log.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallSet.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/transforms/host_runtime/runtime_passes.h"
#include "machina/compiler/mlir/machina/transforms/host_runtime/tpu_metadata_utils.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/compiler/mlir/machina/utils/convert_type.h"
#include "machina/compiler/mlir/machina/utils/device_util.h"
#include "machina/compiler/mlir/machina/utils/dynamic_shape_utils.h"
#include "machina/compiler/mlir/machina/utils/parallel_execute_util.h"
#include "machina/compiler/mlir/machina/utils/serialize_mlir_module_utils.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"
#include "machina/compiler/mlir/machina/utils/xla_rewrite_util.h"
#include "machina/compiler/mlir/machina/utils/xla_sharding_util.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "machina/xla/xla.pb.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

constexpr char kStepMarkerLocationAttr[] = "step_marker_location";
constexpr char kDevicesAttr[] = "devices";
constexpr char kVersionsAttr[] = "tf.versions";
constexpr char kUseXlaSpmdAttr[] = "use_spmd_for_xla_partitioning";

constexpr char kBadStringArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not a string";
constexpr char kBadArrayElementMsg[] =
    "bad '{0}' attribute at index {1} with value '{2}': failed to parse to {3}";
constexpr char kBadArrayAttrLengthMsg[] =
    "bad '{0}' attribute, expected array attribute of size {1}, got size {2}";

namespace {

#define GEN_PASS_DEF_TPUREWRITEPASS
#include "machina/compiler/mlir/machina/transforms/host_runtime/runtime_passes.h.inc"

struct TPURewritePass : public impl::TPURewritePassBase<TPURewritePass> {
  explicit TPURewritePass(toolchain::StringRef _module_name)
      : module_name(_module_name) {}

  void runOnOperation() override;

  toolchain::StringRef module_name;
};

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(toolchain::StringRef attribute) {
  return toolchain::formatv("requires attribute '{0}'", attribute).str();
}

LogicalResult EncapsulateFuncAndSerialize(const std::string& module_name,
                                          func::FuncOp entry_func,
                                          std::string* serialized_func_module) {
  ModuleOp module = entry_func->getParentOfType<ModuleOp>();
  SymbolTable entry_module_table(module);
  toolchain::SmallVector<func::FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningOpRef<mlir::ModuleOp> module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()),
                       absl::StrCat("module_", module_name));
  auto parent_module = entry_func->getParentOfType<ModuleOp>();
  auto versions_attr = parent_module->getAttr(kVersionsAttr);
  if (!versions_attr)
    return parent_module.emitError(CreateMissingAttributeMsg(kVersionsAttr));

  module_for_func.get().getOperation()->setAttr(kVersionsAttr, versions_attr);
  SymbolTable symbol_table(module_for_func.get());

  while (!referenced.empty()) {
    auto func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (symbol_table.lookup<func::FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(func);
    assert(uses && "expected to be able to collect symbol uses");
    for (SymbolTable::SymbolUse use : *uses) {
      func::FuncOp referenced_func = entry_module_table.lookup<func::FuncOp>(
          mlir::cast<FlatSymbolRefAttr>(use.getSymbolRef()).getValue());

      // Skip Symbols that do not map to a function.
      if (!referenced_func) continue;

      referenced.emplace_back(referenced_func);
    }

    auto clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      // We can simply change name of TPU program's main function because there
      // should be no other reference to it.
      clone.setName(StringAttr::get(clone.getContext(), "main"));
      clone.setPublic();
    } else {
      clone.setPrivate();
    }
    symbol_table.insert(clone);
  }

  *serialized_func_module =
      machina::SerializeMlirModule(module_for_func.get());
  return success();
}

// Create a `tf._TPUCompileMlir` that contains a MLIR module that is
// functionally equivalent to the function referenced by cluster_func.
Operation* BuildCompileOp(
    toolchain::StringRef module_name, tf_device::ClusterFuncOp cluster_func,
    int num_replicas, int num_cores_per_replica,
    toolchain::StringRef compilation_device,
    std::optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    OpBuilder* builder, bool tpu_compile_metadata_debug) {
  // Set metadata from attributes.
  machina::tpu::TPUCompileMetadataProto metadata;
  if (!module_name.empty()) metadata.set_module_name(module_name.str());
  if (failed(mlir::TFTPU::SetMetadataProtoFromClusterFuncOp(
          cluster_func, num_replicas, num_cores_per_replica,
          std::move(xla_device_assignment), &metadata)))
    return nullptr;

  // Build a shape op for each input to cluster_func.
  // TODO(b/139377366): When shape inference is ready, we can use compile time
  // shape inference to get inputs that have static shapes and only use shape
  // ops for the rest.
  toolchain::SmallVector<Value, 4> compile_op_operands;
  compile_op_operands.reserve(cluster_func.getNumOperands());

  for (auto operand_and_idx : toolchain::enumerate(cluster_func.getOperands())) {
    // Skip adding shape op for operands that have static shapes.
    machina::PartialTensorShape shape(
        metadata.args(operand_and_idx.index()).shape());
    if (shape.IsFullyDefined()) continue;

    VLOG(1) << "Building compile op for module_name: " << module_name.str()
            << " dynamic shape for operand index: " << operand_and_idx.index()
            << " metadata: "
            << metadata.args(operand_and_idx.index()).DebugString();

    auto shape_op = builder->create<TF::ShapeOp>(
        cluster_func.getLoc(),
        machina::GetTypeFromTFTensorShape({-1}, builder->getIntegerType(64)),
        operand_and_idx.value());
    compile_op_operands.emplace_back(shape_op.getResult());
  }

  FlatSymbolRefAttr func_attr = cluster_func.getFuncAttr();
  func::FuncOp func =
      cluster_func->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
          func_attr.getValue());

  std::string txt_module;
  if (failed(EncapsulateFuncAndSerialize(
          module_name.empty() ? "unknown_graph" : module_name.str(), func,
          &txt_module)))
    return nullptr;

  auto compilation_status_type =
      RankedTensorType::get({}, builder->getType<TF::StringType>());
  auto program_type =
      RankedTensorType::get({3}, builder->getType<TF::StringType>());

  // Add MLIR module's fingerprint to compile metadata.
  uint64_t mlir_fingerprint = machina::Fingerprint64(txt_module);
  metadata.set_mlir_fingerprint(mlir_fingerprint);

  std::string txt_metadata;
  if (tpu_compile_metadata_debug) {
    ::machina::protobuf::TextFormat::Printer printer;
    printer.SetExpandAny(true);
    printer.PrintToString(metadata, &txt_metadata);
  } else {
    metadata.SerializeToString(&txt_metadata);
  }

  auto compile_op = builder->create<TF::_TPUCompileMlirOp>(
      cluster_func.getLoc(),
      /*compilation_status=*/compilation_status_type, /*program=*/
      toolchain::SmallVector<Type, 8>(num_cores_per_replica, program_type),
      compile_op_operands, txt_module, txt_metadata);

  return machina::WrapOpInLaunch(builder, compile_op.getLoc(), compile_op,
                                    compilation_device);
}

// Assigns explicit devices to replicate op. An aliased device is created per
// core, and all replica devices per core are grouped together.
void AssignDevicesToReplicate(
    tf_device::ReplicateOp replicate,
    toolchain::ArrayRef<toolchain::SmallVector<machina::TPUDeviceAndHost, 8>>
        tpu_devices,
    OpBuilder* builder) {
  if (!replicate) return;

  const int num_replicas = tpu_devices.size();
  const int num_cores_per_replica = tpu_devices.front().size();

  toolchain::SmallVector<NamedAttribute, 8> device_attrs;
  for (int core = 0; core < num_cores_per_replica; ++core) {
    toolchain::SmallVector<StringRef, 8> devices_by_core;
    devices_by_core.reserve(num_replicas);
    toolchain::SmallVector<StringRef, 8> hosts_by_core;
    hosts_by_core.reserve(num_replicas);
    for (int replica = 0; replica < num_replicas; ++replica) {
      devices_by_core.push_back(tpu_devices[replica][core].device);
      hosts_by_core.push_back(tpu_devices[replica][core].host);
    }

    device_attrs.push_back(
        builder->getNamedAttr(machina::GetDeviceAliasForLogicalCore(core),
                              builder->getStrArrayAttr(devices_by_core)));

    // For data parallelism, also add replicated host devices, as these are
    // necessary for outside compilation.
    device_attrs.push_back(builder->getNamedAttr(
        machina::GetDeviceAliasForHostOfLogicalCore(core),
        builder->getStrArrayAttr(hosts_by_core)));
  }

  replicate->setAttr(kDevicesAttr, builder->getDictionaryAttr(device_attrs));
}

// Creates a `tf.TPUExecute` op that executes TPU program.
LogicalResult BuildExecuteOp(
    const int core_id, toolchain::ArrayRef<xla::OpSharding> output_sharding_config,
    toolchain::ArrayRef<Value> inputs, tf_device::ClusterFuncOp cluster_func,
    OpBuilder* builder, TF::TPUExecuteOp* execute_op) {
  // TODO(b/139377366): Need to snapshot all resource variable inputs in
  // follow-up CLs.
  toolchain::SmallVector<Type, 4> output_types;
  toolchain::SmallVector<int, 4> cluster_to_core_index;
  auto result = machina::GetOutputTypesForLogicalDeviceComputation(
      core_id, output_sharding_config, cluster_func, &output_types,
      &cluster_to_core_index);
  if (failed(result)) return failure();

  // TPUExecute has same output types as cluster_func.
  *execute_op = builder->create<TF::TPUExecuteOp>(cluster_func.getLoc(),
                                                  output_types, inputs);
  auto producer_name_attr = cluster_func->getAttr("_producer_name");
  if (producer_name_attr)
    (*execute_op)->setAttr("_producer_name", producer_name_attr);
  return success();
}

// Given a `ParallelExecute`, replace it with a new `ParallelExecute`. The
// new `ParallelExecute` will replace the child that contains the
// `ClusterFunc` with `num_cores_per_replica` children. It keep other children
// the same. Return values from the child with the `ClusterFunc` will be
// duplicated `num_cores_per_replica` times.
LogicalResult AddToParallelExecuteOp(
    toolchain::ArrayRef<toolchain::SmallVector<machina::TPUDeviceAndHost, 8>>
        tpu_devices,
    toolchain::ArrayRef<xla::OpSharding> output_sharding_config,
    toolchain::SmallVectorImpl<toolchain::SmallVector<int, 4>>* cluster_to_core_index,
    int num_results_pre_cluster, Operation* compile_op,
    tf_device::ClusterFuncOp cluster_func, OpBuilder* builder,
    tf_device::ParallelExecuteOp old_parallel_execute,
    tf_device::ParallelExecuteOp* new_parallel_execute, int* cluster_idx) {
  const int num_cores_per_replica = tpu_devices.front().size();
  // parallel_execute op returns concatenated list of return values of
  // all its regions.
  //
  // TODO(b/149102702): Correctly map inputs to parallel_execute op via
  // identifying xla_sharding op in the cluster_func function.
  const auto cluster_result_types = cluster_func.getResultTypes();
  toolchain::SmallVector<Type, 8> concatenated_output_types;
  concatenated_output_types.reserve(num_results_pre_cluster +
                                    cluster_result_types.size() *
                                        num_cores_per_replica);
  for (mlir::Region& region : old_parallel_execute.getRegions()) {
    if (!toolchain::isa<tf_device::ClusterFuncOp>(region.front().front())) {
      for (Type t : region.front().front().getResultTypes())
        concatenated_output_types.emplace_back(t);
    }
  }

  for (int core = 0; core < num_cores_per_replica; ++core) {
    cluster_to_core_index->emplace_back(toolchain::SmallVector<int, 4>());
    toolchain::SmallVector<Type, 4> output_types;
    auto result = machina::GetOutputTypesForLogicalDeviceComputation(
        core, output_sharding_config, cluster_func, &output_types,
        &(*cluster_to_core_index)[core]);
    if (failed(result)) return failure();

    for (Type t : output_types) concatenated_output_types.emplace_back(t);
  }

  *cluster_idx = machina::MovePreservedParallelExecuteChildren(
      num_cores_per_replica, concatenated_output_types, builder, cluster_func,
      old_parallel_execute, new_parallel_execute);

  // Extract inputs for each block of the parallel_execute op. The i-th
  // element in the list represents the input lists to TPU computation for
  // i-th logical core.
  toolchain::SmallVector<toolchain::SmallVector<mlir::Value, 4>, 4> input_list;
  builder->setInsertionPoint(*new_parallel_execute);
  auto result = machina::ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func, builder, &input_list);
  if (failed(result)) return failure();

  const bool replicated = tpu_devices.size() != 1;
  // For each logical core, create a region with TPUExecute op.
  assert(input_list.size() == num_cores_per_replica);
  for (int core = 0; core < num_cores_per_replica; ++core) {
    auto& block =
        new_parallel_execute->GetRegionBlockWithIndex((*cluster_idx) + core);
    builder->setInsertionPointToEnd(&block);

    // Create Execute op.
    //
    // TODO(b/148913294): Identify inputs/return values specific to each
    // logical core TPU execution by parsing xla_sharding op in
    // cluster_func.
    auto execute_inputs = input_list[core];
    execute_inputs.emplace_back(compile_op->getResult(core + 1));

    TF::TPUExecuteOp execute;
    result = BuildExecuteOp(core, output_sharding_config, execute_inputs,
                            cluster_func, builder, &execute);
    if (failed(result)) return failure();

    // If computation is replicated, use aliased device. Otherwise there is only
    // one execution device per core and the device is assigned to the execute
    // op.
    std::string device = replicated
                             ? machina::GetDeviceAliasForLogicalCore(core)
                             : tpu_devices.front()[core].device;
    auto block_launch_op = machina::WrapOpInLaunch(
        builder, block.getParent()->getLoc(), execute, device);

    builder->create<tf_device::ReturnOp>(block.getParent()->getLoc(),
                                         block_launch_op.getResults());
  }

  return success();
}

// Creates a `tf.TPUCompileSucceededAssert` operation that parses compilation
// status of `compile_op` to check whether compilation is successful.
void BuildTPUCompileSucceededAssertOp(Operation* compile_op,
                                      Operation* result_id,
                                      toolchain::StringRef compilation_device,
                                      OpBuilder* builder) {
  auto assert_op = builder->create<TF::TPUCompileSucceededAssertOp>(
      compile_op->getLoc(), result_id->getResult(0));
  machina::WrapOpInLaunch(builder, compile_op->getLoc(), assert_op,
                             compilation_device);
}

LogicalResult CheckTPUPartitionedInputAndOutputAreValid(
    tf_device::ClusterFuncOp cluster,
    tf_device::ParallelExecuteOp parallel_execute) {
  for (auto cluster_result : parallel_execute.getExecuteOutputs()) {
    for (Operation* user :
         toolchain::make_early_inc_range(cluster_result.getUsers())) {
      // Check that user has no outputs that are TPUPartitionedOutputV2
      for (auto result : user->getResults()) {
        for (Operation* user : toolchain::make_early_inc_range(result.getUsers())) {
          if (toolchain::isa<TF::TPUPartitionedOutputV2Op>(user)) {
            user->emitError() << "Input of TPUPartitionedOutputV2 must "
                              << "be in tpu computation.";
            return failure();
          }
        }
      }
    }
  }
  for (auto cluster_operand : cluster.getOperands()) {
    Operation* def = cluster_operand.getDefiningOp();
    // This pass assumes that a TPUPartitionedInputV2 is preceeded by
    // ReadVariable ops, and not vice versa. An earlier pass,
    // TPUResourceReadsWritesPartitioning, should have ensured this
    // precondition.
    if (!def) continue;
    for (auto operand : def->getOperands()) {
      Operation* def_of_read = operand.getDefiningOp();
      if (toolchain::isa_and_nonnull<TF::TPUPartitionedInputV2Op>(def_of_read)) {
        def_of_read->emitError() << "Output of TPUPartitionedInputV2 must "
                                 << "be in tpu computation.";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult CheckParallelExecuteConstainsValidNonClusterProcess(
    tf_device::ParallelExecuteOp parallel_execute) {
  int num_pre_cluster_regions = 0;
  int num_post_cluster_regions = 0;
  int num_cluster_regions = 0;
  for (mlir::Region& region : parallel_execute.getRegions()) {
    if (toolchain::isa<tf_device::LaunchFuncOp>(region.front().front())) {
      if (num_cluster_regions == 0) {
        num_pre_cluster_regions++;
      } else {
        num_post_cluster_regions++;
      }
    } else {
      num_cluster_regions++;
    }
  }
  if (num_post_cluster_regions > 0) {
    return failure();
  }
  if (num_pre_cluster_regions > 2) {
    return failure();
  }
  return success();
}

int GetNumResultsPreCluster(tf_device::ParallelExecuteOp parallel_execute) {
  int num_results_pre_cluster = 0;
  for (mlir::Region& region : parallel_execute.getRegions()) {
    if (toolchain::isa<tf_device::LaunchOp>(region.front().front())) {
      num_results_pre_cluster = region.front().front().getResultTypes().size();
    }
  }
  return num_results_pre_cluster;
}

LogicalResult Rewrite(
    toolchain::StringRef module_name, tf_device::ClusterFuncOp cluster_func,
    toolchain::ArrayRef<machina::DeviceNameUtils::ParsedName> devices,
    ArrayRef<TF::TPUCompilationResultOp> compilation_result, OpBuilder* builder,
    bool tpu_compile_metadata_debug) {
  // Fetch the ParallelExecute parent of `cluster_func`, or create it if it does
  // not exist.
  tf_device::ParallelExecuteOp old_parallel_execute =
      cluster_func->getParentOfType<tf_device::ParallelExecuteOp>();
  if (old_parallel_execute &&
      cluster_func->getParentOp() != old_parallel_execute) {
    cluster_func->emitError() << "The ParallelExecute ancestor of a "
                                 "ClusterFunc must be its direct parent.";
    return failure();
  }
  if (!old_parallel_execute)
    old_parallel_execute = TF::BuildParallelExecuteOp(cluster_func, builder);

  // check TPUPartitionedInputV2 and TPUPartitionedOutputV2 are in valid pattern
  if (failed(CheckTPUPartitionedInputAndOutputAreValid(cluster_func,
                                                       old_parallel_execute)))
    return failure();

  if (failed(CheckParallelExecuteConstainsValidNonClusterProcess(
          old_parallel_execute))) {
    old_parallel_execute.emitError()
        << "contains invalid number of non TPU Process";
    return failure();
  }

  // After outside compilation the host process can return results, which come
  // before the cluster_func's results. Collect the number of the outputs from
  // those non cluster_func op
  int num_results_pre_cluster = GetNumResultsPreCluster(old_parallel_execute);

  // Collect `num_replicas` and `num_cores_per_replica` attributes.
  int num_replicas = 1;
  tf_device::ReplicateOp replicate =
      cluster_func->getParentOfType<tf_device::ReplicateOp>();
  if (replicate) num_replicas = replicate.getN();

  auto num_cores_per_replica_attr = cluster_func->getAttrOfType<IntegerAttr>(
      machina::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(machina::kNumCoresPerReplicaAttr));

  int num_cores_per_replica = num_cores_per_replica_attr.getInt();

  auto topology_attr =
      cluster_func->getAttrOfType<StringAttr>(machina::kTopologyAttr);
  if (!topology_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(machina::kTopologyAttr));

  auto device_assignment_attr = cluster_func->getAttrOfType<mlir::ArrayAttr>(
      machina::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster_func.emitOpError(
        toolchain::formatv("requires attribute '{0}'",
                      machina::kDeviceAssignmentAttr)
            .str());

  auto status_or_device_coodinates =
      machina::GetDeviceCoordinates(device_assignment_attr);
  if (!status_or_device_coodinates.ok())
    return cluster_func.emitError()
           << "error in fetching tpu device coordinates: "
           << status_or_device_coodinates.status().message();

  // Determine compilation and execution devices.
  auto status_or_tpu_device_assignment =
      machina::GetTPUCompilationAndExecutionDevices(
          devices, num_replicas, num_cores_per_replica,
          topology_attr.getValue(), status_or_device_coodinates.value());
  if (!status_or_tpu_device_assignment.ok())
    return cluster_func.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().message();

  // Create compile op.
  auto& tpu_device_assignment = status_or_tpu_device_assignment.value();

  // Create the TPUCompileMlir and TPUCompileSucceededAssert outside of
  // the parallel_execute.
  builder->setInsertionPoint(old_parallel_execute);
  Operation* compile_op = BuildCompileOp(
      module_name, cluster_func, num_replicas, num_cores_per_replica,
      tpu_device_assignment.compilation_device,
      std::move(tpu_device_assignment.xla_device_assignment), builder,
      tpu_compile_metadata_debug);
  if (!compile_op) return failure();

  // This replaces _TPUCompileMlir placeholder ops that are required
  // by XlaRecvAtHost and XlaSendFromHost ops add in earlier pass.
  // TODO(b/157054714): When a better abstraction instead of _TPUCompileMlirOp
  // and _XlaRecvAtHostOp and _XlaSendFromHostOp are used, update to a more
  // structured lowering.
  old_parallel_execute.walk(
      [&](TF::_XlaCompileMlirPlaceholderProgramKeyOp key_op) {
        key_op.replaceAllUsesWith(compile_op->getResult(1));
        key_op.erase();
      });

  // After rewrite, if there is a TPUCompilationResultOp from the same cluster,
  // replace it with the result of the compile op. The TPUCompilationResultOp is
  // used as a placeholder to hook during graph creation the other ops that are
  // intended to consume the compile result.
  Operation* result_id = compile_op;
  // TODO(jpienaar): Remove this later.
  auto compile_device_op = compile_op->getAttr("device");
  for (auto res : compilation_result) {
    // Build identity op with the same location/name as the original compilation
    // result op.
    result_id = builder->create<TF::IdentityOp>(
        res.getLoc(), compile_op->getResult(0).getType(),
        result_id->getResult(0));
    // Assign to same device as result is currently set, unless unset and then
    // assign to the device on which compilation will happen.
    // TODO(jpienaar): Remove this later.
    if (auto device = res->getAttrOfType<StringAttr>("device")) {
      if (!device.getValue().empty())
        result_id->setAttr("device", device);
      else
        result_id->setAttr("device", compile_device_op);
    } else if (compile_device_op) {
      result_id->setAttr("device", compile_device_op);
    }
    res.getOutput().replaceAllUsesWith(compile_op->getResult(0));
  }

  BuildTPUCompileSucceededAssertOp(
      compile_op, result_id, tpu_device_assignment.compilation_device, builder);

  AssignDevicesToReplicate(replicate, tpu_device_assignment.tpu_devices,
                           builder);

  toolchain::SmallVector<xla::OpSharding, 4> output_shardings;
  auto result = machina::ParseAndValidateOutputSharding(
      num_cores_per_replica, cluster_func, &output_shardings);
  if (failed(result)) return failure();

  // For model parallelism, mlir::tf_device.parallel_execute is used to express
  // concurrent device execution across multiple logical devices.
  tf_device::ParallelExecuteOp new_parallel_execute;
  int cluster_idx;
  toolchain::SmallVector<toolchain::SmallVector<int, 4>, 4> cluster_to_core_index;
  cluster_to_core_index.reserve(num_cores_per_replica);
  result = AddToParallelExecuteOp(
      tpu_device_assignment.tpu_devices, output_shardings,
      &cluster_to_core_index, num_results_pre_cluster, compile_op, cluster_func,
      builder, old_parallel_execute, &new_parallel_execute, &cluster_idx);
  if (failed(result)) return failure();

  // As mlir::tf_device.parallel_execute wraps # logical cores number of
  // TPUExecute ops, the number of return values of parallel_execute op exceeds
  // that of cluster_func op. As such, each return value of parallel_execute op
  // must be mapped with corresponding return value usages of cluster_func.
  result = machina::RemapOutputsFromLogicalDevices(
      cluster_func.getLoc(), output_shardings, cluster_to_core_index,
      num_results_pre_cluster, old_parallel_execute, cluster_idx,
      new_parallel_execute, builder);
  if (failed(result)) return failure();

  return TF::RemoveSingletonParallelExecuteOp(new_parallel_execute, builder);
}

void TPURewritePass::runOnOperation() {
  TF::RuntimeDevices devices;
  if (failed(machina::GetDevicesFromOp(getOperation(), &devices)))
    return signalPassFailure();

  // Collect compilation results.
  toolchain::DenseMap<Attribute, SmallVector<TF::TPUCompilationResultOp, 1>>
      compilation_results;
  auto result_init = getOperation().walk([&](TF::TPUCompilationResultOp op) {
    auto cluster_id = op->getAttrOfType<StringAttr>("_tpu_compilation_status");
    if (!cluster_id) {
      op->emitOpError("missing '_tpu_compilation_status'");
      return WalkResult::interrupt();
    }
    compilation_results[cluster_id].push_back(op);
    return WalkResult::advance();
  });
  if (result_init.wasInterrupted()) return signalPassFailure();
  toolchain::SmallVector<tf_device::ClusterFuncOp> to_be_erased;
  OpBuilder builder(&getContext());
  auto result = getOperation().walk([&](tf_device::ClusterFuncOp op) {
    if (failed(TF::HasValidCompilationAndReplicationAttributes(*op)))
      return WalkResult::interrupt();
    // Skip non-tpu device cluster_func.
    auto cluster_id = op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
    if (!cluster_id) return WalkResult::advance();

    if (failed(Rewrite(module_name, op, devices.device_names(),
                       compilation_results[cluster_id], &builder,
                       tpu_compile_metadata_debug_)))
      return WalkResult::interrupt();

    to_be_erased.push_back(op);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();

  if (failed(machina::EraseClusterFuncs(to_be_erased)))
    return signalPassFailure();

  // Eliminate TPUCompilationResultOp now that the rewrite is complete.
  for (auto& it : compilation_results) {
    for (auto op : it.second) {
      if (!op.use_empty()) {
        mlir::InFlightDiagnostic err = op.emitError("uses remain post rewrite");
        for (auto user : op->getUsers())
          err.attachNote(user->getLoc()) << "remaining user";
        return signalPassFailure();
      }
      op.erase();
    }
  }

  // TODO(b/139377366): Remove functions that are no longer needed.
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPURewritePass(
    toolchain::StringRef module_name) {
  return std::make_unique<TPURewritePass>(module_name);
}

}  // namespace TFTPU
}  // namespace mlir
