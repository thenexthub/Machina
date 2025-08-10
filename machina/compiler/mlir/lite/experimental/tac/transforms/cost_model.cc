/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/compiler/mlir/lite/experimental/tac/transforms/cost_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/experimental/tac/common/cost.h"
#include "machina/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "machina/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "machina/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// These are just fake costs.
constexpr float kDequantCost = 2.0;
constexpr float kQuantCost = 2.0;
constexpr float kRequantCost = 2.0;

// TODO(renjieliu): Ideally this should consider different kinds of SOCs as
// well.

// Get total bytes transferred.
int64_t GetTransferredTensorBytes(func::CallOp from_graph,
                                  func::CallOp to_graph) {
  int64_t total_size_transferred = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type =
          mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      // Quantized type does not support getSizeInBits.
      if (IsQUI8Type(input_type) || IsQI8Type(input_type)) {
        total_size_transferred += input_type.getNumElements() * 8;
      } else {
        auto s_type = mlir::cast<ShapedType>(input_type);
        total_size_transferred +=
            s_type.getNumElements() * s_type.getElementTypeBitWidth();
      }
    }
  }
  return total_size_transferred;
}

// Get total tensor element size transferred.
int64_t GetTransferredElementCount(func::CallOp from_graph,
                                   func::CallOp to_graph) {
  int64_t total_element_count = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type =
          mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      total_element_count += input_type.getNumElements();
    }
  }
  return total_element_count;
}

struct GetOpCostPass
    : mlir::PassWrapper<GetOpCostPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GetOpCostPass)

  toolchain::StringRef getArgument() const final { return "tfl-get-op-cost"; }
  toolchain::StringRef getDescription() const final {
    return "Get cost for every op";
  }
  void runOnOperation() override;
};

void GetOpCostPass::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func);
  func.walk([&](Operation* op) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !toolchain::isa<func::ReturnOp, func::FuncOp, CallOpInterface>(op)) {
      auto hardware = GetTargetAnnotation(op);
      if (!hardware) return;
      float cost = GetCostForOp(op, *hardware);
      UpdateCost(op, cost, &builder);
    }
  });
}

}  // namespace

float GetCostForOp(Operation* op, const std::string& hardware) {
  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetOpCost(op);
}

float GetCostForFunc(func::FuncOp* func, const std::string& hardware) {
  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetFuncCost(func);
}

float GetTransferCost(const std::string& from_hardware_str,
                      const std::string& to_hardware_str,
                      func::CallOp from_graph, func::CallOp to_graph) {
  auto from_hardware = GetTargetHardware(from_hardware_str);
  auto to_hardware = GetTargetHardware(to_hardware_str);
  if (from_hardware == nullptr) {
    from_graph.emitError(absl::StrCat(
        "we cannot find the registered hardware: ", from_hardware_str));
  }

  if (to_hardware == nullptr) {
    to_graph.emitError(absl::StrCat("we cannot find the registered hardware: ",
                                    to_hardware_str));
  }

  const int64_t total_size_transferred =
      GetTransferredTensorBytes(from_graph, to_graph);
  return to_hardware->GetHardwareSwitchingCost(from_hardware,
                                               total_size_transferred);
}

float GetQuantDequantCost(InferenceType from_inference_type,
                          InferenceType to_inference_type,
                          func::CallOp from_graph, func::CallOp to_graph) {
  // Same inference type, no dequant/quant happens.
  if (from_inference_type == to_inference_type) return 0;

  const int64_t total_element_count_transferred =
      GetTransferredElementCount(from_graph, to_graph);

  if (from_inference_type == FLOAT || from_inference_type == HYBRID) {
    // FLOAT <-> HYBRID will have no quant/dequant as well.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return 0;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // QUANT path.
      return kQuantCost * total_element_count_transferred;
    }
  }

  if (from_inference_type == QUANTIZED_INT8 ||
      from_inference_type == QUANTIZED_UINT8) {
    // Dequant path.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return kDequantCost * total_element_count_transferred;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // Requant path.
      return kRequantCost * total_element_count_transferred;
    }
  }

  // Default quant/dequant/requant cost.
  return kDefaultFixedValuedCost;
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateGetOpCostPass() {
  return std::make_unique<GetOpCostPass>();
}

static PassRegistration<GetOpCostPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
