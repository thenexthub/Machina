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

#include "machina/core/ir/interfaces.h"

#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Region.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/ir/ops.h"
#include "machina/core/ir/types/dialect.h"

namespace mlir {
namespace tfg {

LogicalResult ControlArgumentInterface::verifyRegion(Operation *op,
                                                     Region &region) {
  unsigned num_ctl = 0, num_data = 0;
  for (BlockArgument arg : region.getArguments()) {
    bool is_ctl = mlir::isa<tf_type::ControlType>(arg.getType());
    num_ctl += is_ctl;
    num_data += !is_ctl;
  }
  if (num_ctl != num_data) {
    return op->emitOpError("region #")
           << region.getRegionNumber()
           << " expected same number of data values and control tokens ("
           << num_data << " vs. " << num_ctl << ")";
  }
  return success();
}

void StatefulMemoryEffectInterface::getEffects(
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) const {
  auto registry = dyn_cast<TensorFlowRegistryInterface>(op);
  // If the registry interface is not available, conservatively assume stateful.
  // Otherwise, add a write effect if the operation is known to be stateful.
  // FIXME: Prevent ops in GraphOp being pruned. Remove this when GraphToFunc
  // and FuncToGraph land.
  if (!registry || registry.isStateful() || op->getParentOfType<GraphOp>()) {
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

}  // namespace tfg
}  // namespace mlir

// Include the generated definitions.
#include "machina/core/ir/interfaces.cc.inc"
