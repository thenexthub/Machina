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
#include <vector>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/UseDefLists.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// Checks if a function only contains a tf_executor.graph.
bool IsSupportedGraph(func::FuncOp func) {
  if (!toolchain::hasSingleElement(func)) return false;

  Block& block = func.front();
  if (!toolchain::hasSingleElement(block.without_terminator())) return false;

  auto graph = toolchain::dyn_cast<tf_executor::GraphOp>(block.front());
  if (!graph) return false;

  Operation* terminator = block.getTerminator();
  if (graph.getNumResults() != terminator->getNumOperands()) return false;
  for (auto result : toolchain::zip(graph.getResults(), terminator->getOperands()))
    if (std::get<0>(result) != std::get<1>(result)) return false;

  return true;
}

// Checks if an operation of the tf_executor dialect can have TPU devices
// propagated through.
bool IsSupportedExecutorOp(Operation& op) {
  auto ops_have_same_device = [](Operation* lhs, Operation* rhs) {
    auto lhs_device_attr = lhs->getAttrOfType<StringAttr>(kDeviceAttr);
    auto rhs_device_attr = rhs->getAttrOfType<StringAttr>(kDeviceAttr);
    return (!lhs_device_attr && !rhs_device_attr) ||
           (lhs_device_attr && rhs_device_attr &&
            lhs_device_attr.getValue() == rhs_device_attr.getValue());
  };

  // Check if tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink
  // pair has matching devices or no devices.
  if (auto source = toolchain::dyn_cast<tf_executor::NextIterationSourceOp>(op)) {
    return ops_have_same_device(source, source.GetSink());
  } else if (auto sink = toolchain::dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    return ops_have_same_device(sink.GetSource(), sink);
  }

  return toolchain::isa<tf_executor::EnterOp, tf_executor::ExitOp,
                   tf_executor::IslandOp, tf_executor::MergeOp,
                   tf_executor::SwitchOp>(op);
}

// Assigns all data results to a specified device.
void PopulateDeviceForOpResults(
    Operation& op, toolchain::StringRef device,
    toolchain::DenseMap<Value, toolchain::StringRef>& value_to_device) {
  Operation* op_to_update = &op;
  // Use tf_executor.island op if present as non v1 control flow op results are
  // forwarded by a parent tf_executor.island op.
  if (toolchain::isa<tf_executor::IslandOp>(op_to_update->getParentOp()))
    op_to_update = op_to_update->getParentOp();

  for (Value result : op_to_update->getResults()) {
    if (mlir::isa<tf_executor::TokenType>(result.getType())) continue;
    if (mlir::isa<tf_executor::ControlType>(result.getType())) break;

    value_to_device.insert({result, device});
  }
}

// Checks if an operation can have TPU devices propagated through.
bool IsSupportedOpToSetDevice(Operation& op) {
  return IsSupportedExecutorOp(op) ||
         isa<TF::IdentityOp, TF::IdentityNOp, TF::ShapeOp>(op);
}

// Finds nonconflicting TPU device for an operation from its operands. If an
// operand has no device or a non TPU device, or if there are conflicting
// devices, and empty StringRef will be returned. Control dependencies,
// NextIteration.Source -> NextIteration.Sink token dependencies, and
// LoopCond -> Switch data dependencies are ignored.
toolchain::StringRef FindDeviceFromOperands(
    Operation& op,
    const toolchain::DenseMap<Value, toolchain::StringRef>& value_to_device) {
  toolchain::StringRef new_device;
  const bool is_switch = toolchain::isa<tf_executor::SwitchOp>(op);
  for (Value operand : op.getOperands()) {
    if (mlir::isa<tf_executor::TokenType>(operand.getType())) continue;
    if (mlir::isa<tf_executor::ControlType>(operand.getType())) break;

    if (is_switch &&
        toolchain::isa_and_nonnull<tf_executor::LoopCondOp>(operand.getDefiningOp()))
      continue;

    auto it = value_to_device.find(operand);
    if (it == value_to_device.end()) return toolchain::StringRef();

    if (new_device.empty()) {
      new_device = it->getSecond();
      continue;
    }

    if (new_device != it->getSecond()) return toolchain::StringRef();
  }

  return new_device;
}

// Propagates devices from function arguments.
void PropagateDevicesFromArguments(
    func::FuncOp func,
    toolchain::DenseMap<Value, toolchain::StringRef>& value_to_device) {
  for (BlockArgument& arg : func.getArguments()) {
    auto arg_device_attr =
        func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), kFuncDeviceAttr);
    if (!arg_device_attr || arg_device_attr.getValue().empty() ||
        !machina::IsTPUDevice(arg_device_attr.getValue()))
      continue;
    value_to_device.insert({arg, arg_device_attr.getValue()});
  }
}

// Propagates devices from operation operands to results. Updating the device of
// a tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink will result
// in multiple passes over the tf_executor.graph to propagate devices in loops.
void PropagateDevicesInGraph(
    tf_executor::GraphOp graph,
    toolchain::DenseMap<Value, toolchain::StringRef>& value_to_device) {
  auto ops = graph.GetBody().without_terminator();

  bool updated_next_iteration = false;
  do {
    updated_next_iteration = false;
    for (Operation& op : ops) {
      if (!IsSupportedExecutorOp(op)) continue;

      Operation* op_to_update = &op;
      // Unpack inner op of tf_executor.island.
      if (auto island_op =
              toolchain::dyn_cast<tf_executor::IslandOp>(op_to_update)) {
        if (!island_op.WrapsSingleOp()) continue;
        op_to_update = &island_op.GetBody().front();
      }

      // If op already has a TPU device set, simply propagate its device.
      auto device_attr = op_to_update->getAttrOfType<StringAttr>(kDeviceAttr);
      const bool has_device = device_attr && !device_attr.getValue().empty();
      if (has_device && machina::IsTPUDevice(device_attr.getValue())) {
        PopulateDeviceForOpResults(*op_to_update, device_attr.getValue(),
                                   value_to_device);
        continue;
      }

      // Op has an unsupported device.
      if (has_device) continue;

      if (!IsSupportedOpToSetDevice(*op_to_update)) continue;

      toolchain::StringRef new_device =
          FindDeviceFromOperands(*op_to_update, value_to_device);
      if (new_device.empty()) continue;

      auto new_device_attr =
          mlir::StringAttr::get(op_to_update->getContext(), new_device);
      op_to_update->setAttr(kDeviceAttr, new_device_attr);
      PopulateDeviceForOpResults(*op_to_update, new_device_attr.getValue(),
                                 value_to_device);

      if (auto sink =
              toolchain::dyn_cast<tf_executor::NextIterationSinkOp>(op_to_update)) {
        auto source = sink.GetSource();
        source->setAttr(kDeviceAttr, new_device_attr);
        PopulateDeviceForOpResults(*source, new_device_attr.getValue(),
                                   value_to_device);
        updated_next_iteration = true;
      }
    }
  } while (updated_next_iteration);
}

// Propagates devices to function results.
void PropagateDevicesToResults(
    func::FuncOp func, tf_executor::FetchOp fetch,
    const toolchain::DenseMap<Value, toolchain::StringRef>& value_to_device) {
  // We apply all result attributes at once to avoid excessive allocations when
  // we have many result values.
  toolchain::SmallVector<std::vector<NamedAttribute>, 8> result_attrs;
  {
    toolchain::SmallVector<DictionaryAttr, 8> tmp;
    func.getAllResultAttrs(tmp);

    for (const auto& res : tmp) {
      result_attrs.push_back(res.getValue().vec());
    }
  }

  mlir::Builder builder(func.getOperation());

  for (OpOperand& operand : fetch.getOperation()->getOpOperands()) {
    if (mlir::isa<tf_executor::ControlType>(operand.get().getType())) break;
    auto it = value_to_device.find(operand.get());
    if (it != value_to_device.end()) {
      auto device_attr = func.getResultAttrOfType<StringAttr>(
          operand.getOperandNumber(), kFuncDeviceAttr);
      if (device_attr && !device_attr.getValue().empty()) continue;

      // Update the existing attribute named `kFuncDeviceAttr` if found.
      // Otherwise introduce a new attribute.
      auto& resultAttrForOp = result_attrs[operand.getOperandNumber()];
      bool found = false;
      for (int i = 0; i < resultAttrForOp.size(); ++i) {
        auto attr = resultAttrForOp[i];
        if (attr.getName() == kFuncDeviceAttr) {
          resultAttrForOp[i] = builder.getNamedAttr(
              kFuncDeviceAttr,
              StringAttr::get(func.getContext(), it->getSecond()));
          found = true;
          break;
        }
      }
      if (!found) {
        resultAttrForOp.push_back(builder.getNamedAttr(
            kFuncDeviceAttr,
            StringAttr::get(func.getContext(), it->getSecond())));
      }
    }
  }

  toolchain::SmallVector<DictionaryAttr, 8> tmp;
  for (const auto& res : result_attrs) {
    tmp.push_back(builder.getDictionaryAttr(res));
  }
  func.setAllResultAttrs(tmp);
}

#define GEN_PASS_DEF_TPUDEVICEPROPAGATIONPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct TPUDevicePropagation
    : public impl::TPUDevicePropagationPassBase<TPUDevicePropagation> {
  void runOnOperation() override;
};

void TPUDevicePropagation::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!IsSupportedGraph(func)) return;

  toolchain::DenseMap<Value, toolchain::StringRef> value_to_device;
  PropagateDevicesFromArguments(func, value_to_device);
  auto graph = toolchain::cast<tf_executor::GraphOp>(func.front().front());
  PropagateDevicesInGraph(graph, value_to_device);
  PropagateDevicesToResults(func, graph.GetFetch(), value_to_device);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTPUDevicePropagationPass() {
  return std::make_unique<TPUDevicePropagation>();
}

}  // namespace TFTPU
}  // namespace mlir
