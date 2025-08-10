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

#include <cstdint>
#include <memory>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/UseDefLists.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/side_effect_analysis.h"
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"
#include "machina/compiler/mlir/machina/utils/xla_sharding_util.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kDeviceCPU[] = "CPU";
constexpr char kFuncDeviceAttr[] = "tf.device";

struct TPUDynamicLayoutPass
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TPUDynamicLayoutPass, TF::ResourceAliasAnalysis> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TPUDynamicLayoutPass)

  void runOnFunction(
      func::FuncOp func,
      const TF::ResourceAliasAnalysis::Info& resource_alias_analysis);

  StringRef getArgument() const final { return "tf-tpu-dynamic-layout-pass"; }

  StringRef getDescription() const final {
    return "Inserts TPU layout ops to determine layout at run time.";
  }
};

// Checks if the input producer op is supported in this transform. Right now, we
// only check if it is a tf.IteratorGetNext where resource input is coming from
// a VarHandle on CPU or a function argument assigned to CPU.
bool IsSupportedInputOp(
    Operation* op,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  TF::IteratorGetNextOp iterator_op = toolchain::dyn_cast<TF::IteratorGetNextOp>(op);
  if (!iterator_op) return false;

  Value resource_iterator = iterator_op.getIterator();

  if (resource_alias_analysis.IsUnknownResource(resource_iterator))
    return false;
  toolchain::SmallSetVector<Value, 8> aliases =
      resource_alias_analysis.GetResourceAliases(resource_iterator);

  auto is_generator = [](Value val) {
    if (mlir::isa<BlockArgument>(val)) return true;
    Operation* definition = val.getDefiningOp();
    return definition->getNumOperands() == 0 &&
           definition->getNumResults() == 1;
  };

  // Check all generator aliases (ops or function argument) are on CPU.
  func::FuncOp func = iterator_op->getParentOfType<func::FuncOp>();
  return toolchain::all_of(aliases, [&](Value alias) {
    // Ignore non-generator aliases.
    if (!is_generator(alias)) return true;

    StringAttr device;
    if (auto arg = mlir::dyn_cast<BlockArgument>(alias)) {
      device = func.getArgAttrOfType<mlir::StringAttr>(arg.getArgNumber(),
                                                       kFuncDeviceAttr);
    } else {
      device = alias.getDefiningOp()->getAttrOfType<StringAttr>(kDeviceAttr);
    }

    if (!device) return false;
    machina::DeviceNameUtils::ParsedName parsed_device;
    if (!machina::DeviceNameUtils::ParseFullName(device.getValue().str(),
                                                    &parsed_device)) {
      return false;
    }
    return parsed_device.has_type && parsed_device.type == kDeviceCPU;
  });
}

OpBuilder CreateBuilderAfterOp(Operation* op) {
  return OpBuilder(op->getBlock(), ++Block::iterator(op));
}

// Builds a TPUGetLayoutOp with the given compile op and input index.
TF::TPUGetLayoutOp BuildGetLayout(const int64_t execute_arg_index,
                                  Value compilation_key,
                                  tf_device::LaunchOp compile_launch,
                                  OpBuilder* builder) {
  return builder->create<TF::TPUGetLayoutOp>(
      compile_launch.getLoc(),
      toolchain::ArrayRef<Type>{RankedTensorType::get({ShapedType::kDynamic},
                                                 builder->getIntegerType(64))},
      toolchain::ArrayRef<Value>{compilation_key},
      toolchain::ArrayRef<NamedAttribute>{
          builder->getNamedAttr("index",
                                builder->getI64IntegerAttr(execute_arg_index)),
          builder->getNamedAttr("is_output", builder->getBoolAttr(false))});
}

// Builds a TPUCopyWithLayoutOp with the given get_layout op and input.
TF::TPUCopyWithLayoutOp BuildCopyWithLayout(tf_device::LaunchOp execute_launch,
                                            tf_device::LaunchOp compile_launch,
                                            TF::TPUGetLayoutOp get_layout,
                                            Value input, OpBuilder* builder) {
  return builder->create<TF::TPUCopyWithLayoutOp>(
      execute_launch.getLoc(), toolchain::ArrayRef<Type>{input.getType()},
      toolchain::ArrayRef<Value>{input, get_layout.getLayout()});
}

// Performs transformation for a non-replicated input.
void HandleInput(Value input, const int64_t execute_arg_index,
                 TF::TPUExecuteOp execute, tf_device::LaunchOp execute_launch,
                 tf_device::LaunchOp compile_launch) {
  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, execute.getKey(),
                                   compile_launch, &builder);
  builder.setInsertionPoint(execute_launch);
  auto copy_with_layout = BuildCopyWithLayout(execute_launch, compile_launch,
                                              get_layout, input, &builder);
  copy_with_layout->setAttr(kDeviceAttr, execute_launch.getDeviceAttr());
  execute.setOperand(execute_arg_index, copy_with_layout);
}

// Performs transformation for replicated inputs. Returns true if this is a
// supported case (thus transform happened).
bool HandleReplicatedInputs(
    const int64_t execute_arg_index, Value compilation_key,
    tf_device::LaunchOp execute_launch, tf_device::LaunchOp compile_launch,
    mlir::BlockArgument replicate_arg, tf_device::ReplicateOp replicate,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  // We need to know the devices to copy to.
  if (!replicate.getDevices()) return false;

  MutableArrayRef<OpOperand> inputs =
      replicate.GetOperandsForBlockArgument(replicate_arg);
  for (const auto& entry : toolchain::enumerate(inputs)) {
    auto input_op = entry.value().get().getDefiningOp();
    if (!input_op || !IsSupportedInputOp(input_op, resource_alias_analysis))
      return false;
  }
  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, compilation_key,
                                   compile_launch, &builder);
  builder.setInsertionPoint(replicate);
  for (const auto& entry : toolchain::enumerate(inputs)) {
    auto copy_with_layout =
        BuildCopyWithLayout(execute_launch, compile_launch, get_layout,
                            entry.value().get(), &builder);

    auto device_list = mlir::cast<ArrayAttr>(
        replicate.getDevices().value().get(execute_launch.getDevice()));
    copy_with_layout->setAttr(kDeviceAttr,
                              device_list.getValue()[entry.index()]);

    entry.value().set(copy_with_layout);
  }
  return true;
}

// Performs transformation on a compile and associated execute(s) ops. The
// compile should not have other uses.
void HandleCompileAndExecutes(
    tf_device::LaunchOp compile_launch,
    toolchain::MutableArrayRef<tf_device::LaunchOp> execute_launches,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  auto compile =
      toolchain::cast<TF::_TPUCompileMlirOp>(compile_launch.GetBody().front());
  machina::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(compile.getMetadata().str());
  toolchain::SmallVector<toolchain::SmallVector<int64_t, 4>, 4> input_mappings =
      machina::GetMetadataArgumentMapping(metadata);

  bool metadata_updated = false;
  auto maybe_replicate =
      execute_launches.front()->getParentOfType<tf_device::ReplicateOp>();

  for (auto execute_and_input_mapping :
       toolchain::zip(execute_launches, input_mappings)) {
    auto& execute_launch = std::get<0>(execute_and_input_mapping);
    auto execute =
        toolchain::cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
    const auto& input_mapping = std::get<1>(execute_and_input_mapping);

    for (const auto& input_and_idx : toolchain::enumerate(execute.getArgs())) {
      Value input = input_and_idx.value();
      const int64_t execute_arg_index = input_and_idx.index();
      if (auto block_arg = mlir::dyn_cast<BlockArgument>(input)) {
        // For a block argument, consider transforms only when it is a
        // replicated input (defining ops will be outside the replicate node).
        if (maybe_replicate != block_arg.getParentRegion()->getParentOp() ||
            !HandleReplicatedInputs(execute_arg_index, execute.getKey(),
                                    execute_launch, compile_launch, block_arg,
                                    maybe_replicate, resource_alias_analysis)) {
          continue;
        }
      } else {
        // For an op output, consider transforms only when 1) there is no
        // replication or 2) it is outside the replicate node that encloses the
        // execute node. (Because if the op is inside replicate, it is probably
        // not on the host.)
        auto* input_op = input.getDefiningOp();
        if (maybe_replicate &&
            maybe_replicate.getBody().isAncestor(input_op->getParentRegion())) {
          continue;
        }
        if (!IsSupportedInputOp(input_op, resource_alias_analysis)) continue;
        HandleInput(input, execute_arg_index, execute, execute_launch,
                    compile_launch);
      }

      metadata.mutable_args(input_mapping[execute_arg_index])
          ->set_unrestricted_layout(true);
      metadata_updated = true;
    }
  }

  if (metadata_updated)
    compile->setAttr("metadata", StringAttr::get(compile.getContext(),
                                                 metadata.SerializeAsString()));
}

void TPUDynamicLayoutPass::runOnFunction(
    func::FuncOp func,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  func.walk([&](TF::_TPUCompileMlirOp compile) {
    // Detect tf._TPUCompileMlir -> tf.TPUExecute(s).
    auto compile_launch =
        toolchain::dyn_cast<tf_device::LaunchOp>(compile->getParentOp());
    if (!compile_launch || !compile_launch.WrapsSingleOp()) return;

    toolchain::SmallVector<tf_device::LaunchOp, 4> execute_launches;
    execute_launches.reserve(compile_launch.getNumResults() - 1);
    for (Value program_result :
         toolchain::drop_begin(compile_launch.getResults(), 1)) {
      if (!program_result.hasOneUse()) return;
      Operation* user = *program_result.user_begin();
      auto execute = toolchain::dyn_cast<TF::TPUExecuteOp>(user);
      if (!execute) return;
      auto execute_launch =
          toolchain::dyn_cast<tf_device::LaunchOp>(execute->getParentOp());
      if (!execute_launch || !execute_launch.WrapsSingleOp()) return;
      execute_launches.push_back(execute_launch);
    }

    HandleCompileAndExecutes(compile_launch, execute_launches,
                             resource_alias_analysis);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUDynamicLayoutPass() {
  return std::make_unique<TPUDynamicLayoutPass>();
}

}  // namespace TFTPU
}  // namespace mlir
