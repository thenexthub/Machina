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

#include "machina/compiler/mlir/tf2xla/internal/passes/tpu_validate_inputs_utils.h"

#include <string>

#include "absl/strings/match.h"
#include "toolchain/ADT/DenseSet.h"
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain

namespace machina {
namespace tf2xla {
namespace internal {

bool IsPotentialUnsupportedOp(Operation* op) {
  static auto* ops = [] {
    toolchain::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new toolchain::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<InfeedDequeueTupleOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;

  bool is_in_ops = ops->count(abstractOp->getTypeID()) != 0;
  if (!is_in_ops) return false;

  std::string device = "";
  if (!op->hasAttr(kDeviceAttr)) return false;
  device = op->getAttrOfType<StringAttr>(kDeviceAttr).str();
  if (!absl::StrContains(device, kTpuReplicatedCoreZeroAttr)) return false;
  op->emitWarning("TPU_REPLICATED_CORE:0 device is not supported for op = ")
      << op->getName() << " in TF2XLA MLIR Bridge";

  return true;
}

bool HasV1ControlFlow(GraphOp graph) {
  for (Operation& op : graph.GetBody().without_terminator()) {
    auto island_op = toolchain::dyn_cast<mlir::tf_executor::IslandOp>(op);
    if (!island_op) {
      op.emitWarning() << " is v1 control flow op which is not supported in "
                          "TF2XLA MLIR Bridge.";
      return true;
    }
  }
  return false;
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina
